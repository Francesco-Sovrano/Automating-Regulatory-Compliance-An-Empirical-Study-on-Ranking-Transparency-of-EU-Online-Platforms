import os
import json
from quansx.model_building.pipeline import pipeline
from quansx.utils.levenshtein_lib import remove_similar_labels
from more_itertools import unique_everseen
import logging

from quansx.utils.cache_lib import load_or_create_cache, get_iter_uid

logger = logging.getLogger('quansx')

class QAExtractor:

	def __init__(self, options_dict=None):
		if options_dict is None:
			options_dict = {}
		self.model_type = options_dict.get('model_type', 'distilt5')
		self.model_data = options_dict.get('model_data', 'disco-qaamr')
		self.models_dir = options_dict.get('models_dir', './data/models')
		self.use_cuda = options_dict.get('use_cuda', False)
		model_name = f"{self.model_type}-{self.model_data}-multi"
		self.question_generator = pipeline("question-answer-generation", model=os.path.join(self.models_dir,model_name), use_cuda=self.use_cuda)
		self.generate_kwargs = options_dict.get('generate_kwargs', {
			"max_length": 128,
			"num_beams": 10,
			# "length_penalty": 1.5,
			# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
			"early_stopping": True,
			"num_return_sequences": 1,
		})
		self.e2e_generate_kwargs = options_dict.get('e2e_generate_kwargs', {
			"max_length": 128,
			"num_beams": 5,
			# "length_penalty": 1.5,
			# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
			"early_stopping": True,
			"num_return_sequences": 5,
		})
		self.task_list = options_dict.get('task_list', ['answer2question', 'question2answer'])

	@staticmethod
	def e2e_generator_filter_fn(prediction_results):
		# print('before',prediction_results)
		prediction_results = filter(lambda x:x, prediction_results)
		# prediction_results = list(prediction_results)
		# prediction_results = filter(lambda x: len(next(filter(lambda y: x!=y and y in x, prediction_results),[]))==0, prediction_results)
		prediction_results = sorted(prediction_results, key=len) # from smaller to bigger
		prediction_results = remove_similar_labels(prediction_results)
		# print('after',prediction_results)
		return prediction_results

	def convert_question_answer_to_declaration(self, question, answer, key):
		return self.question_generator({
			'task': 'qa2declaration',
			'question': question,
			'answer': answer,
			'key': key, 
			'generate_kwargs': self.generate_kwargs,
			'e2e_generate_kwargs': self.e2e_generate_kwargs,
			'e2e_generator_filter_fn': self.e2e_generator_filter_fn,
		})

	def extract_question_answer_list(self, sentence_list, task_list=None, batch_size=1000, cache_path=None):
		logger.info('Extracting question_answer_dict..')
		question_answer_list = []
		if not task_list:
			task_list = self.task_list

		cache_id = get_iter_uid(sentence_list) if cache_path else None
		for task in task_list:
			for key in tuple(filter(lambda x:x in self.model_data, ['disco','qaamr'])):
				def qg_fn():
					return self.question_generator({
						'task': task,
						'key': key, 
						'context': sentence_list,
						'generate_kwargs': self.generate_kwargs,
						'e2e_generate_kwargs': self.e2e_generate_kwargs,
						'e2e_generator_filter_fn': self.e2e_generator_filter_fn,
						'batch_size': batch_size,
						'cache_path': cache_path,
					}, _flatten=True)
				if cache_id:
					cache_filename = f"{task}_{key}_{batch_size}.extract_question_answer_dict.{cache_id}.pkl"
					partial_qa_list = load_or_create_cache(cache_path+'.'+cache_filename, qg_fn)
				else:
					partial_qa_list = qg_fn()
				question_answer_list += partial_qa_list
		# (qa['question'], qa['answer'], qa['sentence'], task, key)
		# Clean question-answers
		question_answer_iter = filter(lambda x: x['answer'].casefold() not in x['question'].casefold(), question_answer_list) # remove questions containing the answer
		question_answer_iter = unique_everseen(question_answer_iter, key=lambda x: (x['question'].casefold(), x['answer'].casefold(), x['sentence'].casefold()))
		return list(question_answer_iter)
