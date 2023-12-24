import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import nlp
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser


logger = logging.getLogger('quansx')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'multi', 'question_to_declaration_gen', 'question_answer_gen', 'e2e_question_answer_gen', 'e2e_question_gen', 'e2e_answer_gen', 'answer_gen', 'question_gen'. 'multi' means 'question_answer_gen', 'e2e_question_answer_gen' tasks"}, 
    )
    data: str = field(
        metadata={"help": "Which dataset 'disco', 'qaamr', etc.. Select more data at once by joining them with a - (hyphen)."}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="data/squad_multitask",
        metadata={"help": "Path for dataset directory"}, 
    )
    valid_for_question_gen_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the target text"},
    )

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        # if self.model_type == "t5":
        #     dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding=False, # Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
            truncation=True, 
            add_special_tokens=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding=False, # Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
            truncation=True, 
            add_special_tokens=True,
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def filter_answer_gen(example):
    return example['task'].startswith('qa')

def filter_question_gen(example):
    return example['task'].startswith('qg')

def filter_question_answer_gen(example):
    return filter_answer_gen(example) or filter_question_gen(example)

def filter_e2e_answer_gen(example):
    return example['task'].startswith('e2e_answer_gen')

def filter_e2e_question_gen(example):
    return example['task'].startswith('e2e_question_gen')

def filter_e2e_question_answer_gen(example):
    return filter_e2e_answer_gen(example) or filter_e2e_question_gen(example)

def filter_question_to_declaration_gen(example):
    return example['task'].startswith('q2d')

def filter_multi(example):
    return not filter_question_to_declaration_gen(example)


TASK_TO_FILTER_FN = {
    'e2e_question_answer_gen': filter_e2e_question_answer_gen,
    'e2e_question_gen': filter_e2e_question_gen, 
    'e2e_answer_gen': filter_e2e_answer_gen,
    'question_answer_gen': filter_question_answer_gen,
    'answer_gen': filter_answer_gen,
    'question_gen': filter_question_gen,
    'question_to_declaration_gen': filter_question_to_declaration_gen,
    'multi': filter_multi,
}

def filter_disco(example):
    return example['task'].endswith('disco')

def filter_amr(example):
    return example['task'].endswith('qaamr')

def filter_disco_amr(example):
    return filter_disco(example) or filter_amr(example)

DATA_TO_FILTER_FN = {
    'disco': filter_disco,
    'qaamr': filter_amr, 
    'disco-qaamr': filter_disco_amr,
}

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        tokenizer = T5Tokenizer.from_pretrained("bart-base")
    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    train_dataset = nlp.load_dataset(data_args.dataset_path, split=nlp.Split.TRAIN)
    valid_dataset = nlp.load_dataset(data_args.dataset_path, split=nlp.Split.VALIDATION)

    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    train_dataset = train_dataset.filter(DATA_TO_FILTER_FN[data_args.data])
    train_dataset = train_dataset.filter(TASK_TO_FILTER_FN[data_args.task])
    valid_dataset = valid_dataset.filter(DATA_TO_FILTER_FN[data_args.data])
    valid_dataset = valid_dataset.filter(TASK_TO_FILTER_FN[data_args.task])

    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    train_file_name = f"train_data_{data_args.task}_{data_args.model_type}_{data_args.data}.pt"
    train_path = os.path.join(data_args.dataset_path, train_file_name)

    valid_file_name = f"valid_data_{data_args.task}_{data_args.model_type}_{data_args.data}.pt"
    valid_path = os.path.join(data_args.dataset_path, valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = os.path.join(data_args.dataset_path, f"tokenizer_{data_args.task}_{data_args.model_type}_{data_args.data}")
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()

