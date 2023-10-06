from doxpy.misc.doc_reader import DocParser

from doxpy.models.classification.concept_classifier import ConceptClassifier
from doxpy.models.classification.sentence_classifier import SentenceClassifier
from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences

# from doxpy.misc.graph_builder import get_concept_description_dict
from doxpy.misc.levenshtein_lib import remove_similar_labels
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *
from doxpy.models.reasoning import is_not_wh_word

import numpy as np
from collections import Counter
import re
import time
import json
from more_itertools import unique_everseen
import itertools
import wikipedia
from collections import namedtuple
import logging
from doxpy.models.model_manager import ModelManager

ArchetypePertinence = namedtuple('ArchetypePertinence',['archetype','pertinence'])
InformationUnit = namedtuple('InformationUnit',['unit','context'])
# get_information_unit = lambda x: InformationUnit(x['abstract'], x['sentence'])

class AnswerRetrieverBase(ModelManager):
	archetypal_questions_dict = {
		##### Descriptive
		# 'what': 'What is a description of {X}?',
		'what': 'What is {X}?',
		# 'what': 'What is {X}?',
		'who': 'Who is {X}?',
		# 'whom': 'Whom {X}?',
		##### Causal + Justificatory
		'why': 'Why {X}?',
		# 'why-not': 'Why not {X}?',
		##### Counterfactual
		# 'what-if': 'What if {X}?',
		##### Teleological
		# 'what-for': 'What is {X} for?',
		# 'what-for': 'What is {X} for?',
		##### Expository
		'how': 'How is {X}?',
		##### Quantitative
		# 'how-much': 'How much {X}?',
		# 'how-many': 'How many {X}?',
		##### Spatial
		'where': 'Where is {X}?',
		##### Temporal
		'when': 'When is {X}?',
		##### Medium
		# 'who-by': 'Who by {X}?',
		##### Extra
		'which': 'Which {X}?',
		'whose': 'Whose {X}?',
		##### Discourse Relations
		'Expansion.Manner': 'In what manner is {X}?', # (25\%),
		'Contingency.Cause': 'What is the reason for {X}?', # (19\%),
		'Contingency.Effect': 'What is the result of {X}?', # (16\%),
		'Expansion.Level-of-detail': 'What is an example of {X}?', # (11\%),
		'Temporal.Asynchronous.Consequence': 'After what is {X}?', # (7\%),
		'Temporal.Synchronous': 'While what is {X}?', # (6\%),
		'Contingency.Condition': 'In what case is {X}?', # (3),
		'Comparison.Concession': 'Despite what is {X}?', # (3\%),
		'Comparison.Contrast': 'What is contrasted with {X}?', # (2\%),
		'Temporal.Asynchronous.Premise': 'Before what is {X}?', # (2\%),
		'Temporal.Asynchronous.Being': 'Since when is {X}?', # (2\%),
		'Comparison.Similarity': 'What is similar to {X}?', # (1\%),
		'Temporal.Asynchronous.End': 'Until when is {X}?', # (1\%),
		'Expansion.Substitution': 'Instead of what is {X}?', # (1\%),
		'Expansion.Disjunction': 'What is an alternative to {X}?', # ($\leq 1\%$),
		'Expansion.Exception': 'Except when it is {X}?', # ($\leq 1\%$),
		'Contingency.Neg.Cond.': '{X}, unless what?', # ($\leq 1\%$).
	}

	def __init__(self, kg_manager, concept_classifier_options, sentence_classifier_options, betweenness_centrality=None, **args):
		super().__init__(sentence_classifier_options)
		self.disable_spacy_component = ["ner","textcat"]
		
		self.betweenness_centrality = betweenness_centrality
		self.kg_manager = kg_manager

		# Concept classification
		self.concept_classifier_options = concept_classifier_options
		self._concept_classifier = None
		# Sentence classification
		self.sentence_classifier_options = sentence_classifier_options
		self._sentence_classifier = None
		
		self._overview_aspect_set = None
		self._relevant_aspect_set = None

	@property
	def sentence_classifier(self):
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			self._init_sentence_classifier()
		return self._sentence_classifier

	@property
	def concept_classifier(self):
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			self._init_concept_classifier()
		return self._concept_classifier

	@property
	def overview_aspect_set(self):
		if self._overview_aspect_set is None:
			self._overview_aspect_set = set(filter(lambda x: self.kg_manager.is_relevant_aspect(x,ignore_leaves=True), self.kg_manager.aspect_uri_list))
			# Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.
			if self.betweenness_centrality is not None:
				filtered_betweenness_centrality = dict(filter(lambda x: x[-1] > 0, self.betweenness_centrality.items()))
				self._overview_aspect_set &= filtered_betweenness_centrality.keys()
		return self._overview_aspect_set

	@property
	def relevant_aspect_set(self):
		if self._relevant_aspect_set is None:
			self._relevant_aspect_set = set(filter(self.kg_manager.is_relevant_aspect, self.kg_manager.aspect_uri_list))
		return self._relevant_aspect_set

	@property
	def adjacency_list(self):
		return self.kg_manager.adjacency_list

	def _init_sentence_classifier(self):
		self.logger.info('Initialising Sentence Classifier..')
		# Setup Sentence Classifier
		abstract_iter, context_iter, original_triple_iter, source_id_iter = zip(*filter(lambda x: x[0].strip() and x[1].strip(), self.kg_manager.get_sourced_graph()))
		id_doc_iter = tuple(zip(
			zip(original_triple_iter, source_id_iter), # id
			abstract_iter # doc
		))
		self._sentence_classifier.set_documents(id_doc_iter, tuple(context_iter))

	def _init_concept_classifier(self):
		self.logger.info('Initialising Concept Classifier..')
		self._concept_classifier.set_concept_description_dict(self.kg_manager.concept_description_dict)
		self.logger.info(f'This QA is now considering {len(self.kg_manager.aspect_uri_list)} concepts for question-answering.')
	
	def store_cache(self, cache_name):
		super().store_cache(cache_name)
		self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')
		
	def load_cache(self, cache_name, save_if_init=True, **args):
		super().load_cache(cache_name)
		if self._sentence_classifier is None:
			self._sentence_classifier = SentenceClassifier(self.sentence_classifier_options)
			loaded_sentence_classifier = self._sentence_classifier.load_cache(cache_name+'.sentence_classifier.pkl')
			self._init_sentence_classifier()
			if not loaded_sentence_classifier and save_if_init:
				self._sentence_classifier.store_cache(cache_name+'.sentence_classifier.pkl')
		#######
		if self._concept_classifier is None:
			self._concept_classifier = ConceptClassifier(self.concept_classifier_options)
			loaded_concept_classifier = self._concept_classifier.load_cache(cache_name+'.concept_classifier.pkl')
			self._init_concept_classifier()
			if not loaded_concept_classifier and save_if_init:
				self._concept_classifier.store_cache(cache_name+'.concept_classifier.pkl')
		
	@staticmethod
	def get_question_answer_dict_quality(question_answer_dict, top=5):
		return {
			question: {
				# 'confidence': {
				# 	'best': answers[0]['confidence'],
				# 	'top_mean': sum(map(lambda x: x['confidence'], answers[:top]))/top,
				# },
				# 'syntactic_similarity': {
				# 	'best': answers[0]['syntactic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['syntactic_similarity'], answers[:top]))/top,
				# },
				# 'semantic_similarity': {
				# 	'best': answers[0]['semantic_similarity'],
				# 	'top_mean': sum(map(lambda x: x['semantic_similarity'], answers[:top]))/top,
				# },
				'valid_answers_count': len(answers),
				'syntactic_similarity': answers[0]['syntactic_similarity'] if answers else 0,
				'semantic_similarity': answers[0]['semantic_similarity'] if answers else 0,
			}
			for question,answers in question_answer_dict.items()
		}

	@staticmethod
	def get_answer_question_pertinence_dict(question_answer_dict, update_answers=False):
		answer_question_pertinence_dict = {}
		for question,answers in question_answer_dict.items():
			for a in answers:
				question_pertinence_list = answer_question_pertinence_dict.get(a['sentence'],None)
				if question_pertinence_list is None:
					question_pertinence_list = answer_question_pertinence_dict[a['sentence']] = []
				# subject_n_template = QuestionAnswerExtractor.get_question_subject_n_template(question)
				# if subject_n_template:
				# 	question = subject_n_template[-1]#.replace('{X}','').strip(' ?').casefold().replace(' ','_')
				question_pertinence_list.append(ArchetypePertinence(question, a['confidence']))
		if update_answers:
			for question,answers in question_answer_dict.items():
				for a in answers:
					a['question_pertinence_set'] = answer_question_pertinence_dict[a['sentence']]
		return answer_question_pertinence_dict

	@staticmethod
	def merge_duplicated_answers(question_answer_dict):
		# remove answers contained in other answers, replacing them with the longest answers
		valid_answers_list = flatten(question_answer_dict.values())
		valid_answer_sentence_list = list(map(lambda x: x['sentence'], valid_answers_list))
		for question,answers in question_answer_dict.items():
			for x in answers:
				x['sentence'] = max(filter(lambda y: x['sentence'] in y, valid_answer_sentence_list), key=len)
		for question in question_answer_dict.keys():
			question_answer_dict[question] = list(unique_everseen(question_answer_dict[question], key=lambda x: x['sentence']))
		return question_answer_dict

	@staticmethod
	def minimise_question_answer_dict(question_answer_dict):
		AnswerRetrieverBase.logger.info('Minimising question answer dict')
		# remove duplicated answers
		answer_question_dict = AnswerRetrieverBase.get_answer_question_pertinence_dict(question_answer_dict, update_answers=True)
		get_best_answer_archetype = lambda a: max(answer_question_dict[a['sentence']], key=lambda y: y.pertinence).archetype
		return {
			question: list(filter(lambda x: get_best_answer_archetype(x)==question, answers))
			for question,answers in question_answer_dict.items()
		}

	@staticmethod
	def get_question_answer_overlap_dict(question_answer_dict):
		answer_question_dict = AnswerRetrieverBase.get_answer_question_pertinence_dict(question_answer_dict)
		get_question_iter = lambda q,a_list: filter(lambda x: x!=q, (answer_question_pertinence_dict[a['sentence']].archetype for a in a_list))
		return {
			question: Counter(get_question_iter(question,answers))
			for question,answers in question_answer_dict.items()
		}

	def get_answer_relatedness_to_question(self, question_list, answer_list): 
		question_list = list(map(lambda x: x if question.endswith('?') else x+'?', question_list))
		return self.sentence_classifier.get_element_wise_similarity(question_list,answer_list, source_without_context=True, target_without_context=False)
