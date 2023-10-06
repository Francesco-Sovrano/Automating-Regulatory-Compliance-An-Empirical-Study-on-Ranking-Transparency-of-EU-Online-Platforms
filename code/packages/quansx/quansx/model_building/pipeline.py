import itertools
import logging
from typing import Optional, Dict, Union
import re
from more_itertools import unique_everseen
import os
# import numpy as np

from quansx.utils.transformers_lib import preprocess_text
import time
from tqdm import tqdm
import math
import json
from quansx.utils.cache_lib import load_or_create_cache
from quansx.utils.utils import *
import hashlib

import torch
from transformers import(
	AutoModelForSeq2SeqLM, 
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
)

logger = logging.getLogger('quansx')
	
class QuestionAnswerGenerationPipeline:
	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, use_cuda: bool):
		self.model = model
		self.tokenizer = tokenizer

		self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
		self.model.to(self.device)

		assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
		
		if "T5ForConditionalGeneration" in self.model.__class__.__name__:
			self.model_type = "t5"
		else:
			self.model_type = "bart"

	def __call__(self, inputs: Union[Dict, str], _flatten=True):
		logger.info(f"Task: {inputs['task']}")
		cache_path = inputs.get('cache_path',None)
		if cache_path:
			cache_id = int(hashlib.md5(' '.join(inputs["context"]).encode()).hexdigest(), 16)
			cache_filename = f"{inputs['task']}_{inputs['key']}_{inputs['batch_size']}.{cache_id}.pkl"
			cache_path += '.'+cache_filename

		if inputs['task']=='qa2declaration':
			return self.convert_qa_to_declaration(inputs["question"], inputs["answer"], inputs)
		
		if inputs['task']=='answer2question':
			if "answer" in inputs:
				answers = inputs["answer"]
			else:
				answers = self.extract_answers(inputs) if not cache_path else load_or_create_cache(cache_path, lambda: self.extract_answers(inputs))	
			questions = self.generate_question(answers, inputs)
		elif inputs['task']=='question2answer': # question2answer
			if "question" in inputs:
				questions = inputs["question"]
			else:
				questions = self.extract_questions(inputs) if not cache_path else load_or_create_cache(cache_path, lambda: self.extract_questions(inputs))
			# questions = questions[:100]
			answers = self.answer_question(questions, inputs)
		qa_type = (inputs['key'], inputs['task'])
		qa_dict_list = [
			[
				{
					'question': q,
					'answer': a,
					'sentence': s,
					'type': qa_type,
				}
				for q,a in zip(q_list, a_list)
			]
			for s, q_list, a_list in zip(inputs['context'], questions, answers)
		]
		return flatten(qa_dict_list, as_list=True) if _flatten else qa_dict_list

	def answer_question(self, questions, inputs_dict): 
		return self._extract(
			questions, 
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_answer_gen,
			generate_kwargs=inputs_dict.get("generate_kwargs",None),
			batch_size=inputs_dict.get("batch_size",1000),
		)

	def generate_question(self, answers, inputs_dict): 
		return self._extract(
			answers, 
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_question_gen,
			generate_kwargs=inputs_dict.get("generate_kwargs",None),
			batch_size=inputs_dict.get("batch_size",1000),
		)

	def convert_qa_to_declaration(self, questions, answers, inputs_dict):
		return self._extract(
			questions,
			answers, 
			inputs_dict["key"], 
			self._prepare_inputs_for_declaration_gen,
			generate_kwargs=inputs_dict.get("generate_kwargs",None),
			batch_size=inputs_dict.get("batch_size",1000),
		)
	
	def extract_questions(self, inputs_dict): 
		return self._extract_e2e(
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_e2e_question_gen, 
			generate_kwargs=inputs_dict.get("e2e_generate_kwargs",None),
			e2e_generator_filter_fn=inputs_dict.get("e2e_generator_filter_fn",None), # lambda x:x.endswith('?')
			batch_size=inputs_dict.get("batch_size",1000),
		)

	def extract_answers(self, inputs_dict): 
		return self._extract_e2e(
			inputs_dict["context"], 
			inputs_dict["key"], 
			self._prepare_inputs_for_e2e_answer_gen,
			generate_kwargs = inputs_dict.get("e2e_generate_kwargs",None),
			e2e_generator_filter_fn=inputs_dict.get("e2e_generator_filter_fn",None), # lambda x:x.endswith('?')
			batch_size=inputs_dict.get("batch_size",1000),
		)

	def _tokenize(self, inputs):
		return self.tokenizer(
			inputs, 
			# max_length=max_length,
			# padding='max_length', # Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
			padding=True,
			add_special_tokens=True,
			# truncation=True,
			return_tensors="pt"
		)

	def _detokenize(self, outputs):
		return self.tokenizer.batch_decode(
			outputs, 
			skip_special_tokens=True, 
			clean_up_tokenization_spaces=True,
		)
	
	def _prepare_inputs_for_question_gen(self, answer, context, key):
		source_text = f"{key} answer: {answer}  {key} context: {context}"
		# if self.model_type == "t5":
		#	source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_answer_gen(self, question, context, key):
		source_text = f"{key} question: {question}  {key} context: {context}"
		# if self.model_type == "t5":
		#	source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_declaration_gen(self, question, answer, key):
		if not question.endswith('?'): question += '?'
		source_text = f"{key} qa2dec: {' '.join([question,answer])}"
		# if self.model_type == "t5":
		#	source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_e2e_question_gen(self, context, key):
		source_text = f"{key} e2e questions: {context}"
		# if self.model_type == "t5":
		#	source_text = source_text + " </s>"
		return source_text

	def _prepare_inputs_for_e2e_answer_gen(self, context, key):
		source_text = f"{key} e2e answers: {context}"
		# if self.model_type == "t5":
		#	source_text = source_text + " </s>"
		return source_text
	
	def _extract(self, sources, contexts, key, input_formatter_fn, generate_kwargs=None, batch_size=1000):
		context_is_not_list = not isinstance(contexts, (list,tuple))
		if context_is_not_list:
			contexts = [contexts]
		source_is_not_list = not isinstance(sources, (list,tuple))
		if source_is_not_list:
			sources = [sources]

		if generate_kwargs is None:
			generate_kwargs = {}
		# if batch_size > 1:
		# 	if 'num_beam_groups' not in generate_kwargs:
		# 		generate_kwargs['num_beam_groups'] = generate_kwargs.get('num_return_sequences', 1)
		# 	generate_kwargs['num_return_sequences'] = 1 # force it to 1 to avoid severe bug

		logger.info(f'Extracting Q/A of type {key} from {len(contexts)} contexts with batch size {batch_size} and args {json.dumps(generate_kwargs, indent=4)}..')

		def process_chunk(_context_source_list):
			assert len(_context_source_list) <= batch_size, f"en(_context_list) <= batch_size but {len(_context_source_list)} > {batch_size}"
			num_return_sequences = generate_kwargs.get('num_return_sequences', 1)
			# Tokenize sources
			# print(_context_list, _sources_list)
			_sources_text_list = [
				input_formatter_fn(
					preprocess_text(_source), 
					preprocess_text(_context), 
					key
				) 
				for _source,_context in _context_source_list
			]
			# Generate outputs
			tokenized_inputs = self._tokenize(_sources_text_list)
			input_ids=tokenized_inputs['input_ids'].to(self.device)
			attention_mask=tokenized_inputs['attention_mask'].to(self.device)
			_outputs = self.model.generate(
				input_ids=input_ids, 
				attention_mask=attention_mask,
				**generate_kwargs,
			)
			# Decode outputs
			_prediction_results = self._detokenize(_outputs)
			if num_return_sequences > 1:
				_prediction_results = [
					tuple(unique_everseen(flatten(c)))
					for c in get_chunks(_prediction_results, elements_per_chunk=num_return_sequences)
				]
			# Free memory
			if self.device == "cuda":
				del tokenized_inputs['input_ids']
				del tokenized_inputs['attention_mask']
				# torch.cuda.empty_cache()
			# print(result)
			assert len(_context_source_list) == len(_prediction_results)
			return _prediction_results
		
		# Build chunk list
		#####################
		context_sources = [
			(c,s) 
			for c,s_list in zip(contexts,sources) 
			for s in (s_list if s_list else [''])
		]
		chunk_list = tuple(get_chunks(context_sources, elements_per_chunk=batch_size))
		#####################
		# Get results
		results = flatten(map(process_chunk,tqdm(chunk_list)), as_list=True)
		assert len(results) == len(context_sources)
		results_iter = iter(results)
		# Re-group outputs
		return [
			[
				next(results_iter)
				for _ in s_list
			]
			for s_list in sources
		]

	def _extract_e2e(self, contexts, key, input_formatter_fn, generate_kwargs=None, e2e_generator_filter_fn=None, batch_size=1000):
		context_is_not_list = not isinstance(contexts, (list,tuple))
		if context_is_not_list:
			contexts = [contexts]
		if generate_kwargs is None:
			generate_kwargs = {}
		# if batch_size > 1:
		# 	if 'num_beam_groups' not in generate_kwargs:
		# 		generate_kwargs['num_beam_groups'] = generate_kwargs.get('num_return_sequences', 1)
		# 	generate_kwargs['num_return_sequences'] = 1 # force it to 1 to avoid severe bug

		logger.info(f'Extracting end2end Q/A of type {key} from {len(contexts)} contexts with batch size {batch_size} and args {json.dumps(generate_kwargs, indent=4)}..')

		def process_chunk(_context_list):
			assert len(_context_list) <= batch_size, f"en(_context_list) <= batch_size but {len(_context_list)} > {batch_size}"
			num_return_sequences = generate_kwargs.get('num_return_sequences', 1)
			# Tokenize sources
			_source_text_list = [
				input_formatter_fn(preprocess_text(_context), key)
				for _context in _context_list
			]
			tokenized_input = self._tokenize(_source_text_list)
			input_ids = tokenized_input['input_ids'].to(self.device)
			attention_mask = tokenized_input['attention_mask'].to(self.device)
			
			# torch.random.seed()
			_outputs = self.model.generate(
				input_ids=input_ids, 
				attention_mask=attention_mask,
				**generate_kwargs,
			)
			# Decode outputs
			_prediction_results = [
				tuple(unique_everseen(map(preprocess_text, decoded_output.split("<sep>")[:-1])))
				for decoded_output in self._detokenize(_outputs)
			]
			if num_return_sequences > 1:
				_prediction_results = [
					tuple(unique_everseen(flatten(c)))
					for c in get_chunks(_prediction_results, elements_per_chunk=num_return_sequences)
				]
			# print(len(_context_list), len(_source_text_list), len(input_ids), len(_prediction_results), _outputs.shape, len(flatten(_prediction_results,as_list=True)))
			# print('b', json.dumps(list(zip(_context_list, _prediction_results)), indent=4))
			# Free memory
			if self.device == "cuda":
				del tokenized_input['input_ids']
				del tokenized_input['attention_mask']
				# torch.cuda.empty_cache()
			if e2e_generator_filter_fn:
				_prediction_results = [
					e2e_generator_filter_fn(x)
					for x in _prediction_results
				]
			# print(_prediction_results)
			assert len(_prediction_results) == len(_context_list)
			return _prediction_results
		
		# Build chunk list
		#####################
		chunk_list = tuple(get_chunks(contexts, elements_per_chunk=batch_size))
		#####################
		# Get results
		results = flatten(map(process_chunk,tqdm(chunk_list)), as_list=True)
		assert len(results) == len(contexts)
		# if context_is_not_list:
		#   return results[0]
		return results

class QuestionGenerationPipeline(QuestionAnswerGenerationPipeline):
	def __call__(self, inputs: Union[Dict, str]):
		return self.extract_questions(inputs)

class AnswerGenerationPipeline(QuestionAnswerGenerationPipeline):
	def __call__(self, inputs: Union[Dict, str]):
		return self.extract_answers(inputs)

SUPPORTED_TASKS = {
	"question-answer-generation": {
		"impl": QuestionAnswerGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-qa-qg-hl",
		}
	},
	"question-generation": {
		"impl": QuestionGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-e2e-qg",
		}
	},
	"answer-generation": {
		"impl": AnswerGenerationPipeline,
		"default": {
			"model": "valhalla/t5-small-qa-qg-hl",
		}
	},
}

def pipeline(
	task: str,
	model: Optional = None,
	tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
	use_cuda: Optional[bool] = True,
	**kwargs,
):
	# Retrieve the task
	if task not in SUPPORTED_TASKS:
		raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

	targeted_task = SUPPORTED_TASKS[task]
	task_class = targeted_task["impl"]

	# Use default model/config/tokenizer for the task if no model is provided
	if model is None:
		model = targeted_task["default"]["model"]
	
	# Try to infer tokenizer from model or config name (if provided as str)
	if tokenizer is None:
		if isinstance(model, str):
			tokenizer = model
		else:
			# Impossible to guest what is the right tokenizer here
			raise Exception(
				"Impossible to guess which tokenizer to use. "
				"Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
			)
	
	# Instantiate tokenizer if needed
	if isinstance(tokenizer, (str, tuple)):
		if isinstance(tokenizer, tuple):
			# For tuple we have (tokenizer name, {kwargs})
			tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
		else:
			tokenizer = AutoTokenizer.from_pretrained(tokenizer)
	
	# Instantiate model if needed
	if isinstance(model, str):
		model = AutoModelForSeq2SeqLM.from_pretrained(model)
	
	return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
	