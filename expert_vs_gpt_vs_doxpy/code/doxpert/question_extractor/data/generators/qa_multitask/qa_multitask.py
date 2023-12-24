# coding=utf-8
import json
import logging
import os
import re

import nlp
import pandas as pd
from quansx.utils.transformers_lib import preprocess_text

SCRIPT_DIR = os.path.realpath(__file__)
DATASET_DIR = '/home/toor/Desktop/QuAnsX-Trainer/data/datasets'

def qa_amr_generator(path):
	format_string = lambda x: re.sub(r'  +',' ',x.strip())
	with open(path,'r') as f:
		annotation_file_lines = tuple(f.readlines())
	sentence_annotation_list = []
	i = 0
	while i < len(annotation_file_lines):
		i += 1
		context = format_string(annotation_file_lines[i])
		i += 1
		while i < len(annotation_file_lines) and annotation_file_lines[i].strip():
			line = annotation_file_lines[i]
			i += 1
			splitted_line = line.split('?')
			if len(splitted_line) < 2:
				splitted_line = line.split('  ')
			if len(splitted_line) < 2:
				continue
			question = format_string(splitted_line[0])+'?'
			answer = format_string(' '.join(splitted_line[1:])).split('\t')[1]
			# print(context, question, [answer])
			yield context, question, [answer]
		i += 1

def qa_disco_generator(path):
	df = pd.read_table(path, sep='\t')
	for i,row in df.iterrows():
		context = preprocess_text(row["sentence"])
		question = preprocess_text(row["full_question"])
		if not question.endswith('?'):
			question += '?'
		answer = preprocess_text(row["full_answer"])
		yield context, question, [answer]

def qa_squad_generator(path):
	with open(path) as f:
		squad = json.load(f)
	for article in squad["data"]:
		title = article.get("title", "").strip()
		for paragraph in article["paragraphs"]:
			context = paragraph["context"].strip()
			for qa in paragraph["qas"]:
				question = qa["question"].strip()
				if not question.endswith('?'):
					question += '?'
				# id_ = qa["id"]
				answers = [answer["text"].strip() for answer in qa["answers"]]
				yield context, question, answers

class QAMultitask(nlp.GeneratorBasedBuilder):

	QA_DISCO_URL = os.path.join(DATASET_DIR,"QA-Discourse")
	QA_AMR_URL = os.path.join(DATASET_DIR,"QA-AMR")
	QA_SRL_URL = os.path.join(DATASET_DIR,"QA-SRL")
	SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
	URLS_TO_DOWNLOAD = {
		"train": {
			"disco": [
				os.path.join(QA_DISCO_URL,'wikinews_train.tsv'),
				os.path.join(QA_DISCO_URL,'wikipedia_train.tsv'),
			],
			# "squad": [
			# 	os.path.join(SQUAD_URL,"train-v1.1.json"),
			# ],
			"qaamr": [
				os.path.join(QA_AMR_URL,"train.tsv"),
			],
			# "srl": [
			# 	os.path.join(QA_SRL_URL,"train.jsonl"),
			# ],
		},
		"dev": {
			"disco": [
				os.path.join(QA_DISCO_URL,'wikinews_dev.tsv'),
				os.path.join(QA_DISCO_URL,'wikipedia_dev.tsv'),
			],
			# "squad": [
			# 	os.path.join(SQUAD_URL,"dev-v1.1.json"),
			# ],
			"qaamr": [
				os.path.join(QA_AMR_URL,"dev.tsv"),
			],
			# "srl": [
			# 	os.path.join(QA_SRL_URL,"dev.jsonl"),
			# ],
		},
	}
	
	def _info(self):
		return nlp.DatasetInfo(
			features=nlp.Features(
				{
					"source_text": nlp.Value("string"),
					"target_text": nlp.Value("string"),
					"task": nlp.Value("string"),
				}
			),
			# No default supervised_keys (as we have to pass both question
			# and context as input).
			supervised_keys=None,
		)

	def _split_generators(self, dl_manager):
		downloaded_files = dl_manager.download_and_extract(self.URLS_TO_DOWNLOAD)
		return [
			nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath_dict": downloaded_files['train']}),
			nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath_dict": downloaded_files['dev']}),
		]

	def process_answer_gen_text(self, context, question, answer, key):
		source_text = f"{key} question: {question}  {key} context: {context}"
		return {"source_text": source_text, "target_text": answer, "task": f"qa_{key}"}

	def process_question_gen_text(self, context, question, answer, key):
		source_text = f"{key} answer: {answer}  {key} context: {context}"
		return {"source_text": source_text, "target_text": question, "task": f"qg_{key}"}

	def process_e2e_answer_gen(self, context, answers, key):
		source_text = f"{key} e2e answers: {context}"
		target_text = " {sep_token} ".join(answers)
		target_text += " {sep_token}"
		return {"source_text": source_text, "target_text": target_text, "task": f"e2e_answer_gen_{key}"}

	def process_e2e_question_gen(self, context, questions, key):
		source_text = f"{key} e2e questions: {context}"
		target_text = " {sep_token} ".join(questions)
		target_text += " {sep_token}"
		return {"source_text": source_text, "target_text": target_text, "task": f"e2e_question_gen_{key}"}

	def process_question_to_declaration_gen(self, question, answer, declaration, key):
		if not question.endswith('?'): question += '?'
		source_text = f"{key} qa2dec: {' '.join([question,answer])}"
		return {"source_text": source_text, "target_text": declaration, "task": f"q2d_{key}"}
	
	def _generate_examples(self, filepath_dict):
		"""This function returns the examples in the raw (text) form."""
		count = 0
		tasks = ['qa', 'qg', 'e2e_answer_gen', 'e2e_question_gen', 'q2d']
		# min_annotation_count = float('inf')
		# for key, path_list in filepath_dict.items():
		# 	for path in path_list:
		# 		if key == "disco":
		# 			generator = qa_disco_generator
		# 		elif key == "squad":
		# 			generator = qa_squad_generator
		# 		annotation_count = len(list(generator(path)))
		# 		if annotation_count < min_annotation_count:
		# 			min_annotation_count = annotation_count

		for key, path_list in filepath_dict.items():
			# key_count = 0
			for path in path_list:
				logging.info("generating examples from <%s>", path)
				if key == "disco":
					generator = qa_disco_generator
				elif key == "squad":
					generator = qa_squad_generator
				elif key == "qaamr":
					generator = qa_amr_generator
				else:
					raise Exception(f"Unsupported data: {key}")

				context_answers_dict = {}
				context_questions_dict = {}
				for context, question, answers in generator(path):
					# if key_count > min_annotation_count:
					# 	break
					# key_count += 1
					if context not in context_questions_dict:
						context_questions_dict[context] = []
						context_answers_dict[context] = []
					context_answers_dict[context] += answers
					context_questions_dict[context].append(question)
					if 'qa' in tasks:
						yield count, self.process_answer_gen_text(context, question, answers[0], key)
						count += 1
					if 'qg' in tasks:
						yield count, self.process_question_gen_text(context, question, answers[0], key)
						count += 1
					if 'q2d' in tasks:
						yield count, self.process_question_to_declaration_gen(question, answers[0], context, key)
						count += 1

				if 'e2e_question_gen' in tasks:
					for context, questions in context_questions_dict.items():
						yield count, self.process_e2e_question_gen(context, questions, key)
						count += 1
				if 'e2e_answer_gen' in tasks:
					for context, answers in context_answers_dict.items():
						yield count, self.process_e2e_answer_gen(context, answers, key)
						count += 1
