import sys
import json
import numpy as np
import re
from more_itertools import unique_everseen
import math
import os

from doxpy.models.model_manager import ModelManager
from quansx.question_answer_extraction import QAExtractor
from doxpy.misc.levenshtein_lib import labels_are_contained
from doxpy.misc.adjacency_list import AdjacencyList
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.knowledge_extraction.concept_extractor import ConceptExtractor
# from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences
from doxpy.misc.graph_builder import tuplefy, get_subject_set, get_object_set
# from doxpy.misc.doc_reader import clean_content
from doxpy.misc.cache_lib import load_or_create_cache
from doxpy.misc.utils import *

from doxpy.misc.jsonld_lib import *
from doxpy.models.knowledge_extraction.knowledge_graph_manager import singlefy

class QuestionAnswerExtractor(ModelManager):
	question_template_dict = {
		'What {X}?': '{obj} {verb} {subj}',
		'Who {X}?': '{obj} {verb} {subj}',
		'Why {X}?': '{subj} {verb} because {obj}',
		'How {X}?': '{obj} {verb} {subj}',
		'How much {X}?': '{obj} {verb} {subj}',
		'Where {X}?': '{obj} {verb} {subj}',
		'When {X}?': '{obj} {verb} {subj}',
		'Who by {X}?': '{subj} {verb} by {obj}',
		'Which {X}?': '{obj} {verb} {subj}',
		'Whose {X}?': '{subj} {verb} of {obj}',
		##### Discourse Relations
		'In what manner {X}?': '{obj} {verb} {subj}',
		'After what {X}?': '{subj} {verb} after {obj}',
		'While what {X}?': '{subj} {verb} while {obj}',
		'In what case {X}?': '{subj} {verb} when {obj}',
		'Despite what {X}?': '{subj} {verb} despite {obj}',
		'Before what {X}?': '{subj} {verb} before {obj}',
		'Since when {X}?': '{subj} {verb} since {obj}',
		'Until when {X}?': '{subj} {verb} until {obj}',
		'Instead of what {X}?': '{subj} {verb} instead of {obj}',
		'Except when {X}?': '{subj} {verb} except {obj}',
		'Unless what {X}?': '{subj} {verb} unless {obj}',
	}
	question_template_list = sorted(question_template_dict.keys(), key=len, reverse=True)

	def __init__(self, model_options):
		super().__init__(model_options)
		self.disable_spacy_component = ["ner", "textcat"]
		self._qa_extractor = None

	@property
	def qa_extractor(self):
		if not self._qa_extractor:
			self._qa_extractor = QAExtractor(self.model_options)
		return self._qa_extractor

	def extract_aligned_graph_from_qa_dict_list(self, kg_manager, qa_dict_list, graph_builder_options, qa_type_to_use=None, avoid_jumps=False, remove_stopwords=False, remove_numbers=False, parallel_extraction=True, add_verbs=False, add_predicates_label=False, use_paragraph_text=False, elements_per_chunk=None, cache_path=None):
		if qa_type_to_use is not None: # disco or qaamr
			qa_dict_list = list(filter(lambda x: x['type'][0] in qa_type_to_use, qa_dict_list))

		# Set correct paragraph_text and doc before building the EDUs graph
		span_source_id_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
		source_uri_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_PARAGRAPH_ID_PREDICATE)
		if use_paragraph_text:
			source_to_sentence_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
		else:
			source_to_sentence_list = []
			for x,source_span_text_list in kg_manager.adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy).items():
				for source_span_text in source_span_text_list:
					# if 'The law applicable should also govern the question of the capacity to incur liability in tort/delict.' in source_span_text:
					# 	print('ooo',x, source_span_text, source_span_text=='The law applicable should also govern the question of the capacity to incur liability in tort/delict.')
					if x in span_source_id_dict:
						source_to_sentence_list += [
							(source_uri,source_span_text)
							for source_sentence_uri in span_source_id_dict[x]
							for source_uri in source_uri_dict[source_sentence_uri]
						]
					else:
						source_to_sentence_list += [
							(source_uri,source_span_text)
							for source_uri in source_uri_dict[x]
						]
			source_to_sentence_dict = {}
			for source_uri, source_span_text in source_to_sentence_list:
				if source_uri not in source_to_sentence_dict:
					source_to_sentence_dict[source_uri] = []
				source_to_sentence_dict[source_uri].append(source_span_text)
		### Clean memory
		del span_source_id_dict
		del source_uri_dict
		################
		
		# Build content-to-source dict
		content_to_source_uri_dict = {
			sentence: source_uri
			for source_uri,sentence_list in source_to_sentence_dict.items()
			for sentence in sentence_list
		}
		del source_to_sentence_dict

		# Build qa2sentence_dict
		qa2sentence_dict = {}
		for qa_dict in qa_dict_list:
			sentence = qa_dict['sentence']
			source_uri = content_to_source_uri_dict.get(sentence, None)
			# assert source_uri, f"Could not find: {sentence}"
			if not source_uri:
				self.logger.debug(f'<extract_aligned_graph_from_qa_dict_list> "{sentence}" is missing')
				continue
			question_answer = qa_dict['abstract'] # QA extractor is not normalising strings, but KnowledgeGraphExtractor will
			sentence_dict = qa2sentence_dict.get(question_answer, None)
			if sentence_dict is None:
				sentence_dict = qa2sentence_dict[question_answer] = []
			sentence_dict.append((sentence,source_uri))
		del content_to_source_uri_dict
		# print(json.dumps(qa2sentence_dict, indent=4))

		qa_iter = map(lambda x: x['abstract'], qa_dict_list)
		qa_iter = filter(lambda x: x in qa2sentence_dict, qa_iter)
		qa_list = list(qa_iter)
		assert qa_list, f"No valid QA found in qa_dict_list of length {len(qa_dict_list)}"

		########################
		self.logger.info(f'QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Processing {len(qa_list)} QAs..')
		chunks = tuple(get_chunks(qa_list, elements_per_chunk=elements_per_chunk)) if elements_per_chunk else [qa_list]
		self.logger.info(f'QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Processing {len(chunks)} chunks..')
		kg_builder = KnowledgeGraphExtractor(graph_builder_options)
		def process_chunk(contents):
			kg_builder.set_content_list(
				contents, 
				remove_stopwords=remove_stopwords, 
				remove_numbers=remove_numbers, 
				avoid_jumps=avoid_jumps, 
				parallel_extraction=parallel_extraction,
			)
			new_triplet_list = []
			doc_uri_dict = kg_manager.adjacency_list.get_predicate_dict(DOC_ID_PREDICATE)
			content_dict = kg_manager.adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
			for subj_dict, predicate_dict, obj_dict in kg_builder.triplet_tuple:
				qa = predicate_dict['source']['paragraph_text']
				predicate_dict['source_text'] = qa # this will force the OntologyBuilder to consider the EDUs/AMRs as subj/obj instead of triplets
				assert len(qa2sentence_dict[qa]) > 0
				for sentence_text, source_uri in qa2sentence_dict[qa]:
					doc_uri = doc_uri_dict[source_uri][0]#.replace(DOC_PREFIX,'') # Do not remove DOC_PREFIX here
					annotation_content = kg_manager.get_sub_graph(source_uri)
					for paragraph_text in content_dict[source_uri]:
						new_predicate_dict = dict(predicate_dict) # copy
						new_predicate_dict['source'] = dict(predicate_dict['source']) # copy

						new_predicate_dict['source']['sentence_text'] = sentence_text
						# assert len(content_dict[source_uri]) == 1
						new_predicate_dict['source']['paragraph_text'] = paragraph_text
						new_predicate_dict['source']['doc'] = doc_uri
						new_predicate_dict['source']['annotation'] = {
							'root': source_uri, # important to have the same source_uri of the 'source graph', without it graphs alignment might be incomplete
							'content': annotation_content, # add source_id sub-graph
						}
						new_triplet_list.append((subj_dict, new_predicate_dict, obj_dict))
			# assert new_triplet_list, 'new_triplet_list is empty'
			kg_builder.triplet_tuple = tuple(new_triplet_list)
			# Build EDUs graph
			kg_builder.logger.info('QuestionAnswerExtractor::extract_aligned_graph_from_qa_dict_list - Extracting edge_list')
			# edge_list_fn = kg_builder.parallel_get_edge_list if parallel_extraction else kg_builder.get_edge_list
			return kg_builder.get_edge_list(
				kg_builder.triplet_tuple, 
				add_subclasses=True, 
				use_framenet_fe=False, 
				use_wordnet=False, # keep it to false, we are adding wordnet references later and faster
				lemmatize_label=graph_builder_options.get('lemmatize_label', False), 
				add_verbs=add_verbs, 
				add_predicates_label=add_predicates_label, 
			)
		if elements_per_chunk and cache_path:
			for i,c in enumerate(chunks): # create caches consuming less memory
				load_or_create_cache(
					cache_path+f'.extract_aligned_graph_from_qa_dict_list_{i}_{elements_per_chunk}.{get_iter_uid(c)}.pkl',
					lambda: process_chunk(c)
				) if cache_path else process_chunk(c)
			edu_graph = flatten(
				(
					load_or_create_cache(
						cache_path+f'.extract_aligned_graph_from_qa_dict_list_{i}_{elements_per_chunk}.{get_iter_uid(c)}.pkl', 
						lambda: process_chunk(c)
					) if cache_path else process_chunk(c)
					for i,c in enumerate(chunks)
				)
			)
		else:
			edu_graph = flatten(map(process_chunk, chunks))
		edu_graph = tuplefy(unique_everseen(edu_graph))
		########################
		# Add useful sub-class relations
		subclass_dict = kg_manager.adjacency_list.get_predicate_dict(SUBCLASSOF_PREDICATE)
		subclass_graph = [
			(a,SUBCLASSOF_PREDICATE,b)
			for a,b_list in subclass_dict.items()
			for b in b_list
		]
		del subclass_dict
		edu_graph += subclass_graph
		edu_graph += filter(lambda x: '{obj}' not in x[1], unique_everseen(flatten(map(kg_manager.get_sub_graph, get_object_set(subclass_graph)))))
		del subclass_graph
		
		edu_graph = list(unique_everseen(edu_graph))
		return edu_graph

	@staticmethod
	def get_question_subject_n_template(question, sorted_template_list=None): 
		if not question.endswith('?'):
			question = question+' ?'
		# else:
		# 	question.replace('?',' ?')
		if not sorted_template_list:
			sorted_template_list = QuestionAnswerExtractor.question_template_list
		for template in sorted_template_list:
			template_re = re.compile(re.escape(template).replace("\\{X\\}",' *([^ ].*)'), re.IGNORECASE)
			m = re.match(template_re, question)
			if not m:
				continue
			question_subject = m.groups()[0].strip()
			return (question_subject, template) if question_subject else None
		return None

	def convert_interrogative_to_declarative_sentence_list(self, interrogative_sentence_list, paraphraser_options=None):
		if paraphraser_options is None:
			paraphraser_options = {
				# 'truncation': True,
				# 'padding': 'longest',
				'max_length': 60,
				'num_beams': 10, 
				'temperature': 1.5,
			}
		declarative_sentence_list = []
		for interrogative_sentence in interrogative_sentence_list:
			# Extract question and answer from the interrogative sentence
			if interrogative_sentence.count('?') != 1:
				self.logger.warning(f'<convert_interrogative_to_declarative_sentence_list> "{interrogative_sentence}" is an invalid interrogative_sentence')
				declarative_sentence_list.append(None)
				continue
			q,a = interrogative_sentence.split('?')
			q = (q+'?').strip()
			obj = a.strip()
			# Get question and answer from the interrogative sentence
			q_subj_template = self.get_question_subject_n_template(q)
			if not q_subj_template:
				self.logger.warning(f'<convert_interrogative_to_declarative_sentence_list> "{interrogative_sentence}" does not fit any template')
				declarative_sentence_list.append(None)
				continue
			q_subj, q_template = q_subj_template
			qa2decl_fn = lambda s,v,o: self.question_template_dict[q_template].replace('{subj}',s).replace('{verb}',v).replace('{obj}',o)
			# Get declarative sentence analysing the dependency tree of the question
			q_span = self.nlp([q])[0]
			root_t = next(filter(lambda t: t.dep_=='ROOT', q_span))
			# print(root_t, '=', root_t.pos_, root_t.tag_, root_t.dep_)
			if root_t.pos_=='NOUN':
				declarative_sentence = qa2decl_fn(q_subj,'is',obj)
			else:
				auxiliary_list = list(ConceptExtractor.get_token_descendants(root_t, lambda x: x.dep_ not in ConceptExtractor.SUBJ_IDENTIFIER))
				verb_span = sorted(filter(lambda x: x.text not in q_template, [root_t]+auxiliary_list), key=lambda x: x.i)
				verb = ConceptExtractor.get_span_text(verb_span)
				subj_span = tuple(filter(lambda x: x.text not in q_template and x not in verb_span, q_span))
				subj = ConceptExtractor.get_span_text(subj_span)
				declarative_sentence = qa2decl_fn(subj,verb,obj)
			declarative_sentence_list.append(declarative_sentence)
			# print('+', declarative_sentence)
		# Grammatically fix the sentence
		return list(map(lambda x: x['generated_text'] if x else None, self.run_hf_task(
			declarative_sentence_list, 
			num_return_sequences=1,
			**paraphraser_options
		)))

	def clean_qa_dict_list(self, qa_dict_list, sorted_template_list=None, min_qa_pertinence=0.05, max_qa_similarity=0.9502, max_answer_to_question_overlap=0.75, min_answer_to_sentence_overlap=0.75, min_question_to_sentence_overlap=0.75, coreference_resolution=False, cache_path=None):
		tokens_iter = lambda s: filter(lambda y: y, map(lambda x: x.strip(',;:.?!"\'[](){}s'), s.casefold().replace('-',' ').split(' ')))
		def qa_in_sentence_ratio(qa, sentence):
			qa_set = set(tokens_iter(qa))
			qa_in_sentence_set = qa_set.intersection(tokens_iter(sentence))
			return (len(qa_in_sentence_set)/len(qa_set)) if qa_set else -1
		###########################################
		# Remove question with more than 1 question mark
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Removing QAs with more than 1 question mark.')
		qa_dict_list = list(filter(lambda x: x['question'].count('?') <= 1, qa_dict_list))
		###########################################
		# Remove QAs whose question is not in the sentence
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Removing QAs whose answer is not in the sentence.')
		# for x in qa_dict_list:
		# 	if not qa_in_sentence_ratio(x['answer'], x['sentence']) >= min_answer_to_sentence_overlap:
		# 		print("qa_in_sentence_ratio(x['answer'], x['sentence']) >= min_answer_to_sentence_overlap", json.dumps(x, indent=4), qa_in_sentence_ratio(x['answer'], x['sentence']))
		qa_dict_list = list(filter(lambda x: qa_in_sentence_ratio(x['answer'], x['sentence']) >= min_answer_to_sentence_overlap, self.tqdm(qa_dict_list)))
		###########################################
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Cleaning QA dict list..')
		# Clean and format QAs
		# self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Clean and format QA')
		abstract_list = [x['abstract'].strip() for x in qa_dict_list]
		question_list = [x['question'].strip() for x in qa_dict_list]
		question_list = [x if x.endswith('?') else x+'?' for x in question_list]
		answer_list = [x['answer'].strip() for x in qa_dict_list]
		if coreference_resolution:
			self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Remove coreferences from QA')
			clean_question_list_fn = lambda: self.resolve_texts_coreferences(question_list)
			question_list = load_or_create_cache(
				cache_path+f'.question_list.{get_iter_uid(question_list)}.pkl', 
				clean_question_list_fn
			) if cache_path else clean_question_list_fn()
			clean_answer_list_fn = lambda: self.resolve_texts_coreferences(answer_list)
			answer_list = load_or_create_cache(
				cache_path+f'.answer_list.{get_iter_uid(answer_list)}.pkl', 
				clean_answer_list_fn
			) if cache_path else clean_answer_list_fn()
			abstract_list = [' '.join([q,a]) for q,a in zip(question_list,answer_list)]
		for qa_dict, abstract, question, answer in zip(qa_dict_list, abstract_list, question_list, answer_list):
			qa_dict['abstract'] = abstract
			qa_dict['question'] = question
			qa_dict['answer'] = answer
		### Clean memory
		del question_list
		del answer_list
		del abstract_list
		###########################################
		# Remove QAs whose question does not fit any pre-defined template
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Remove QAs whose question does not fit any pre-defined template')
		# for x in qa_dict_list:
		# 	if not self.get_question_subject_n_template(x['question'], sorted_template_list):
		# 		print("self.get_question_subject_n_template(x['question'], sorted_template_list)", json.dumps(x, indent=4), self.get_question_subject_n_template(x['question'], sorted_template_list))
		qa_dict_list = list(filter(lambda x: self.get_question_subject_n_template(x['question'], sorted_template_list), self.tqdm(qa_dict_list)))
		###########################################
		# Remove QAs whose question is not in the sentence
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Removing QAs whose question is not in the sentence..')
		# for x in qa_dict_list:
		# 	if not qa_in_sentence_ratio(self.get_question_subject_n_template(x['question'], sorted_template_list)[0], x['sentence']) >= min_question_to_sentence_overlap:
		# 		print("qa_in_sentence_ratio(self.get_question_subject_n_template(x['question'], sorted_template_list)[0], x['sentence']) >= min_question_to_sentence_overlap", json.dumps(x, indent=4), qa_in_sentence_ratio(self.get_question_subject_n_template(x['question'], sorted_template_list)[0], x['sentence']))
		qa_dict_list = list(filter(lambda x: qa_in_sentence_ratio(self.get_question_subject_n_template(x['question'], sorted_template_list)[0], x['sentence']) >= min_question_to_sentence_overlap, self.tqdm(qa_dict_list)))
		###########################################
		# Remove QAs whose question is the sentence
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Removing QAs whose question is the sentence..')
		# for x in qa_dict_list:
		# 	if not qa_in_sentence_ratio(x['sentence'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) < 1:
		# 		print("qa_in_sentence_ratio(x['sentence'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) < 1", json.dumps(x, indent=4), qa_in_sentence_ratio(x['sentence'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]))
		qa_dict_list = list(filter(lambda x: qa_in_sentence_ratio(x['sentence'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) < 1, self.tqdm(qa_dict_list)))
		###########################################
		# Remove QAs whose answer is the question
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Removing QAs whose answer is the question..')
		# for x in qa_dict_list:
		# 	if not qa_in_sentence_ratio(x['answer'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) <= max_answer_to_question_overlap:
		# 		print("qa_in_sentence_ratio(x['answer'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) <= max_answer_to_question_overlap", json.dumps(x, indent=4), qa_in_sentence_ratio(x['answer'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]))
		qa_dict_list = list(filter(lambda x: qa_in_sentence_ratio(x['answer'], self.get_question_subject_n_template(x['question'], sorted_template_list)[0]) <= max_answer_to_question_overlap, self.tqdm(qa_dict_list)))
		###########################################
		# Remove QAs whose answers are deemed to be not related to the question
		if min_qa_pertinence > 0 or max_qa_similarity < 1:
			abstract_list = [x['abstract'] for x in qa_dict_list]
			question_list = [x['question'] for x in qa_dict_list]
			answer_list = [x['answer'] for x in qa_dict_list]
			self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Remove QAs whose answers are deemed to be not related to the question')
			fn = lambda: self.get_element_wise_similarity(question_list, answer_list, source_without_context=True, target_without_context=False)
			qa_relatedness_list = load_or_create_cache(
				cache_path+f'.qa_relatedness_list.{get_iter_uid(question_list+answer_list)}.pkl', 
				fn
			) if cache_path else fn()
			### Clean memory
			del question_list
			del answer_list
			del abstract_list
		################
		# if only_valid_sentences:
		# 	self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Remove QA of invalid sentences')
		# 	qa_dict_list = filter_invalid_sentences(self, qa_dict_list, key=lambda x: x['sentence'], avoid_coreferencing=False)
		################
		if min_qa_pertinence > 0:
			self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Remove QAs whose answers are deemed to be not related to the question')
			qa_dict_list = [
				x 
				for i,x in enumerate(qa_dict_list) 
				if qa_relatedness_list[i] >= min_qa_pertinence
			]
			qa_relatedness_list = [x for x in qa_relatedness_list if x >= min_qa_pertinence]
			assert len(qa_dict_list)==len(qa_relatedness_list), f'len(qa_dict_list) {len(qa_dict_list)} == len(qa_relatedness_list) {len(qa_relatedness_list)}'
		# Prune the longest similar QA
		if max_qa_similarity < 1:
			self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - Prune the longest similar QA')
			qa_relatedness_dict = {
				qa_dict['abstract']: qa_relatedness
				for qa_dict,qa_relatedness in zip(qa_dict_list,qa_relatedness_list)
			}
			### Clean memory
			del qa_relatedness_list
			################
			# Group QAs by source sentence
			sentence_qa_dict = {}
			for qa_dict in qa_dict_list:
				sentence = qa_dict['sentence']
				x = sentence_qa_dict.get(sentence, None)
				if not x:
					x = sentence_qa_dict[sentence] = []
				x.append(qa_dict)
			sentence_qa_dict = {
				k: sorted(v, key=lambda x: qa_relatedness_dict[x['abstract']], reverse=True)
				for k,v in sentence_qa_dict.items()
			}
			# Remove similar QAs
			self.get_default_embedding(list(qa_relatedness_dict.keys()), without_context=False, with_cache=True) # Fast embedding with cache
			qa_dict_list = [
				qa_dict
				for qa_dict_list in self.tqdm(sentence_qa_dict.values(), total=len(sentence_qa_dict))
				for qa_dict in self.remove_similar_labels(qa_dict_list, threshold=max_qa_similarity, key=lambda x: x['abstract'], with_cache=True)
			]
		self.logger.info('QuestionAnswerExtractor::clean_qa_dict_list - QA dict list cleaned')
		return qa_dict_list

	def extract(self, graph, paraphraser_options=None, add_declarations=False, use_paragraph_text=False, cache_path=None, elements_per_chunk=10000):
		self.logger.info(f'QuestionAnswerExtractor::extract - Extracting QA dict..')
		# Build adjacency matrix from knowledge graph
		adjacency_list = AdjacencyList(
			graph, 
			equivalence_relation_set=set([IS_EQUIVALENT_PREDICATE]),
			is_sorted=True,
		)
		span_source_id_dict = adjacency_list.get_predicate_dict(HAS_SOURCE_ID_PREDICATE)
		if use_paragraph_text:
			content_dict = adjacency_list.get_predicate_dict(HAS_CONTENT_PREDICATE, singlefy)
		else:
			content_dict = adjacency_list.get_predicate_dict(HAS_SOURCE_LABEL_PREDICATE, singlefy)
			content_dict = dict(filter(lambda x: x[0] not in span_source_id_dict, content_dict.items())) # avoid clauses, consider sentences only
		sentence_list = tuple(flatten(content_dict.values()))
		### Clean memory
		del content_dict
		del span_source_id_dict
		del adjacency_list
		################
		# Extract QA dictionary
		qa_dict_list = self.qa_extractor.extract_question_answer_list(sentence_list, batch_size=self.default_batch_size, cache_path=cache_path)
		del sentence_list
		# print(json.dumps(question_answer_matrix, indent=4))
		# Add manually specified QAs
		known_qa_dict = {}
		for qa_id, p, o in filter(lambda x: x[1] in KNOWN_QA_PREDICATES, graph):
			qa = known_qa_dict.get(qa_id, None)
			if qa is None: 
				qa = known_qa_dict[qa_id] = {}
			qa[p] = o
		if known_qa_dict:
			self.logger.debug('QuestionAnswerExtractor::extract - Adding manually specified QA (with tag known_qa):')
			self.logger.debug(json.dumps(known_qa_dict, indent=4))
			qa_dict_list += [
				{
					'question': qa_dict[QUESTION_TEMPLATE_PREDICATE],
					'answer': qa_dict[ANSWER_TEMPLATE_PREDICATE],
					'sentence': qa_dict.get(EXPLANATORY_TEMPLATE_PREDICATE, qa_dict[QUESTION_TEMPLATE_PREDICATE]+' '+qa_dict[ANSWER_TEMPLATE_PREDICATE]),
					'type': ('known_qa','known_qa'),
				}
				for qa_dict in known_qa_dict.values()
				if ANSWER_TEMPLATE_PREDICATE in qa_dict and QUESTION_TEMPLATE_PREDICATE in qa_dict
			]
			del known_qa_dict
		
		# Add missing question marks
		for qa_dict in qa_dict_list:
			qa_dict['question'] = qa_dict['question'].strip().strip('?') + '?'
		# # Remove questions with no subjects
		# self.logger.info(f'QuestionAnswerExtractor::extract - Removing questions with no subject')
		# chunks = tuple(get_chunks(qa_dict_list, elements_per_chunk=elements_per_chunk)) if elements_per_chunk else [qa_dict_list]
		# qa_dict_list = []
		# for chunk in self.tqdm(chunks):
		# 	# has_subj = lambda x: next(filter(lambda y: 'subj' in y, map(lambda x: x.dep_, self.nlp([x])[0])), None) is not None
		# 	question_list = tuple(unique_everseen(map(lambda x: x['question'], chunk)))
		# 	question_nlp_dict = dict(zip(question_list, self.nlp(question_list)))
		# 	del question_list
		# 	has_subj = lambda x: next(filter(lambda y: 'subj' in y, map(lambda x: x.dep_, question_nlp_dict[x])), None) is not None
		# 	qa_dict_list.extend(filter(lambda x: has_subj(x['question']), chunk))
		# 	del question_nlp_dict

		# Add missing abstracts
		for qa_dict in qa_dict_list:
			qa_dict['abstract'] = qa_dict['question'] + ' ' + qa_dict['answer']
		# Add declarations
		if add_declarations:
			abstract_list = list(map(lambda x: x['abstract'], qa_dict_list))
			declaration_list = self.convert_interrogative_to_declarative_sentence_list(abstract_list, paraphraser_options)
			del abstract_list
			for declaration, qa_dict in zip(declaration_list, qa_dict_list):
				qa_dict['declaration'] = declaration
			del declaration_list
		self.logger.info(f'QuestionAnswerExtractor::extract - QA dict extracted')
		return qa_dict_list
