from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from doxpy.models.knowledge_extraction.question_answer_extractor import QuestionAnswerExtractor
from doxpy.misc.doc_reader import load_or_create_cache, DocParser
from doxpy.misc.graph_builder import get_betweenness_centrality, save_graphml, get_concept_set, get_concept_description_dict
from doxpy.models.knowledge_extraction.couple_extractor import filter_invalid_sentences
from doxpy.misc.jsonld_lib import *
from doxpy.misc.utils import *

import json
import os
from os import mkdir, path as os_path
import sys
import re
import time
# import logging
# logger = logging.getLogger('doxpy')
# logger.setLevel(logging.INFO)
# # logger.setLevel(logging.WARNING)
# logger.addHandler(logging.StreamHandler(sys.stdout))

import argparse

parser = argparse.ArgumentParser(description='draw plots')
parser.add_argument('--model_type', dest='model_type', type=str, help='the model type')
parser.add_argument('-p1', '--checklist_pertinence_threshold', dest='checklist_pertinence_threshold', type=float)
parser.add_argument('-p2', '--dox_answer_pertinence_threshold', dest='dox_answer_pertinence_threshold', type=float)
parser.add_argument('-s', '--synonymity_threshold', dest='synonymity_threshold', type=float)
parser.add_argument('--checklist_path', dest='checklist_path', type=str, default=None)
parser.add_argument('--open_question_path', dest='open_question_path', type=str, default=None)
parser.add_argument('--explainable_information_path', dest='explainable_information_path', type=str)
parser.add_argument('--cache_path', dest='cache_path', type=str)
ARGS = parser.parse_args()

model_type, checklist_pertinence_threshold, dox_answer_pertinence_threshold, synonymity_threshold, checklist_path, open_question_path, explainable_information_path, cache_path = ARGS.model_type, ARGS.checklist_pertinence_threshold, ARGS.dox_answer_pertinence_threshold, ARGS.synonymity_threshold, ARGS.checklist_path, ARGS.open_question_path, ARGS.explainable_information_path, ARGS.cache_path
if not os.path.exists(cache_path): os.mkdir(cache_path)

assert checklist_path is not None or open_question_path is not None, "Provide a checklist and/or a set of open questions"

print('Assessing DoX of:', ARGS)

########################################################################################################################################################################
################ Configuration ################
EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES = False # set it to True to reduce the number of considered Q/A and memory footprint
AVOID_JUMPS = True
AVOID_COREFERENCING = False
TEMPERATURE = 0
TOP_P = 0

CHECKLIST_INSTRUCTION = 'It should also start with "Yes" if the answer is positive, "No" if the answer is negative or "N/A" if the answer is not available.'

PROMPT_TEMPLATE = lambda question,contents: f'''Output a comprehensive answer based only and exclusively on the information within the paragraphs below (if any can be used to answer) which were extracted from the documentation to be assessed. If no paragraph can answer the question, then output only "No, I cannot answer". Otherwise, the comprehensive answer must contain citations to the source paragraphs, e.g., blablabla (paragraph 1 and 2), blabla (paragraph 0). {CHECKLIST_INSTRUCTION if checklist_path else ''}

Question:
{question}

Paragraphs:
{contents}
'''


OQA_OPTIONS = {
	'answer_horizon': 20,
	######################
	## AnswerRetriever stuff
	'answer_pertinence_threshold': checklist_pertinence_threshold, 
	'tfidf_importance': 0,

	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.95,
	'use_weak_pointers': False,
	'top_k': 1000,
	# 'filter_fn': lambda a: not ('....' in a['sentence'] or a['sentence'].startswith('*') or a['sentence'].casefold().startswith('figure')),

	'keep_the_n_most_similar_concepts': None, 
	'query_concept_similarity_threshold': .4, 
	'add_external_definitions': False, 
	'include_super_concepts_graph': False, 
	'include_sub_concepts_graph': True, 
	'consider_incoming_relations': True,
	'minimise': False, 
}

ARCHETYPE_FITNESS_OPTIONS = {
	'one_answer_per_sentence': False,
	'answer_pertinence_threshold': dox_answer_pertinence_threshold, 
	'answer_to_question_max_similarity_threshold': None,
	'answer_to_answer_max_similarity_threshold': 0.85,
}

KG_MANAGER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,
	'with_cache': False,
	'with_tqdm': False,

	# 'min_triplet_len': 0,
	# 'max_triplet_len': float('inf'),
	# 'min_sentence_len': 0,
	# 'max_sentence_len': float('inf'),
	# 'min_paragraph_len': 0,
	# 'max_paragraph_len': 0, # do not use paragraphs for computing DoX
}

GRAPH_EXTRACTION_OPTIONS = {
	'add_verbs': False, 
	'add_predicates_label': False, 
	'add_subclasses': True, 
	'use_wordnet': False,
}

GRAPH_CLEANING_OPTIONS = {
	'remove_stopwords': False,
	'remove_numbers': False,
	'avoid_jumps': AVOID_JUMPS,
	'parallel_extraction': False,
}

GRAPH_BUILDER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'with_cache': False,
	'with_tqdm': False,

	'max_syntagma_length': None,
	'add_source': True,
	'add_label': True,
	'lemmatize_label': False,

	# 'default_similarity_threshold': 0.75,
	'default_similarity_threshold': 0,
	'tf_model': {
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5', # Transformer
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
		# 'use_cuda': True,
		# 'with_cache': True,
		# 'batch_size': 100,
	},
}

QA_CLEANING_OPTIONS = {
	# 'sorted_template_list': None, 
	'min_qa_pertinence': 0.05, 
	'max_qa_similarity': 1, 
	'min_answer_to_sentence_overlap': 0.75,
	'min_question_to_sentence_overlap': 0, 
	'max_answer_to_question_overlap': 0.75,
	'coreference_resolution': False,
}

QA_EXTRACTOR_OPTIONS = {
	'models_dir': 'question_extractor/data/models', 

	# 'sbert_model': {
	# 	'url': 'facebook-dpr-question_encoder-multiset-base', # model for paraphrase identification
	# 	'use_cuda': True,
	# },
	'tf_model': {
		# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa2/3', # English QA
		'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
		# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
		# 'use_cuda': True,
	}, 

	# 'with_cache': False,
	'with_tqdm': True,
	'use_cuda': True,
	'default_batch_size': 10,
	'default_cache_dir': cache_path,
	'generate_kwargs': {
		"max_length": 128,
		"num_beams": 10,
		# "num_return_sequences": 1,
		# "length_penalty": 1.5,
		# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
		"early_stopping": True,
	},
	'e2e_generate_kwargs': {
		"max_length": 128,
		"num_beams": 10,
		# "num_beam_groups": 1,
		"num_return_sequences": 10,
		# "length_penalty": 1.5,
		# "no_repeat_ngram_size": 3, # do not set it when answer2question=False, questions always start with the same ngrams 
		"early_stopping": True,
		# "return_dict_in_generate": True,
		# "forced_eos_token_id": True
	},
	'task_list': [
		'answer2question', 
		'question2answer'
	],
}

CONCEPT_CLASSIFIER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	'default_batch_size': 20,
	'with_tqdm':True,
	'with_cache': True,

	'sbert_model': {
		'url': 'all-mpnet-base-v2', # model for paraphrase identification
		# 'use_cuda': True,
		'with_cache': True,
	},
	# 'sbert_model': {
	# 	'url': 'all-MiniLM-L12-v2',
	# 	'use_cuda': True,
	# },
	'default_similarity_threshold': synonymity_threshold,
	
	'default_tfidf_importance': 0,
	'with_stemmed_tfidf': True,
}

SENTENCE_CLASSIFIER_OPTIONS = {
	# 'spacy_model': 'en_core_web_trf',
	# 'n_threads': 1,
	# 'use_cuda': True,

	# 'default_batch_size': 100,
	'with_tqdm': True,
	'with_cache': True,
	
	'default_tfidf_importance': 0,
	'with_stemmed_tfidf': True,
}

################ Initialise data structures ################
def init_qa(graph, model_family='sbert', information_units='edu_amr_clause'):
	with_qa_dict_list = 'edu' in information_units or 'amr' in information_units
	print(f'server_interface {model_family} {information_units}, with with_qa_dict_list: {with_qa_dict_list}')
	# betweenness_centrality_cache = os_path.join(cache_path,'betweenness_centrality.pkl')
	qa_dict_list_cache = os_path.join(cache_path,f'qa_dict_list_{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}.pkl')
	cleaned_qa_dict_list_cache = os_path.join(cache_path,f'cleaned_qa_dict_list_{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}.pkl')
	filtered_qa_dict_list_cache = os_path.join(cache_path,f'filtered_qa_dict_list_{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}.pkl')
	qa_cache = os_path.join(cache_path,f'qa_embedder-{information_units}.pkl')

	################ Configuration ################
	if model_family == 'tf':
		SENTENCE_CLASSIFIER_OPTIONS['tf_model'] = {
			# 'url': 'https://tfhub.dev/google/universal-sentence-encoder-qa/3', # English QA
			'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3', # Multilingual QA # 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
			# 'url': 'https://tfhub.dev/google/LAReQA/mBERT_En_En/1',
			# 'use_cuda': True,
			'with_cache': True,
		}
	else:
		SENTENCE_CLASSIFIER_OPTIONS['sbert_model'] = {
			'url': 'multi-qa-mpnet-base-cos-v1', # model for paraphrase identification
			# 'use_cuda': True,
			'with_cache': True,
		}

	########################################################################
	print('Building Question Answerer..')
	if with_qa_dict_list:
		qa_dict_list = load_or_create_cache(qa_dict_list_cache, lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract(graph, cache_path=qa_dict_list_cache, use_paragraph_text=EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES))
		print(f'qa_dict_list now has len {len(qa_dict_list)}')
		qa_dict_list = load_or_create_cache(cleaned_qa_dict_list_cache, lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).clean_qa_dict_list(qa_dict_list, cache_path=cleaned_qa_dict_list_cache, **QA_CLEANING_OPTIONS))
		print(f'qa_dict_list now has len {len(qa_dict_list)}')
		qa_dict_list = load_or_create_cache(filtered_qa_dict_list_cache, lambda: filter_invalid_sentences(QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS), qa_dict_list, key=lambda x: x['sentence'], avoid_coreferencing=AVOID_COREFERENCING))
		print(f'qa_dict_list now has len {len(qa_dict_list)}')
	else:
		qa_dict_list = None
	edu_graph = []
	if 'edu_amr' in information_units or 'amr_edu' in information_units:
		edu_amr_graph_cache = os_path.join(cache_path,f"graph_{'edu_amr'}_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}_paragraphs-{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}_avoidjumps-{AVOID_JUMPS}.pkl")
		edu_graph = load_or_create_cache(
			edu_amr_graph_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES,
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	elif 'edu' in information_units:
		edu_cache = os_path.join(cache_path,f"graph_{'edu'}_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}_paragraphs-{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}_avoidjumps-{AVOID_JUMPS}.pkl")
		edu_graph = load_or_create_cache(
			edu_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES,
				qa_type_to_use= [
					'disco', # elementary discourse units
					# 'qaamr', # abstract meaning representations
				],
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	elif 'amr' in information_units:
		amr_cache = os_path.join(cache_path,f"graph_{'amr'}_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}_paragraphs-{EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES}_avoidjumps-{AVOID_JUMPS}.pkl")
		edu_graph = load_or_create_cache(
			amr_cache, 
			lambda: QuestionAnswerExtractor(QA_EXTRACTOR_OPTIONS).extract_aligned_graph_from_qa_dict_list(
				KnowledgeGraphManager(KG_MANAGER_OPTIONS, graph), 
				qa_dict_list,
				GRAPH_BUILDER_OPTIONS, 
				use_paragraph_text=EXTRACT_QA_USING_PARAGRAPHS_INSTEAD_OF_SENTENCES,
				qa_type_to_use= [
					# 'disco', # elementary discourse units
					'qaamr', # abstract meaning representations
				],
				**GRAPH_CLEANING_OPTIONS,
			)
		)
	if 'clause' in information_units:
		kg = list(unique_everseen(edu_graph + graph))
		del graph
		# save_graphml(kg, 'knowledge_graph')
		print('Graph size:', len(kg))
		print('Grammatical Clauses:', len(list(filter(lambda x: '{obj}' in x[1], kg))))
		kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, kg)
		del kg
	else:
		del graph
		# save_graphml(edu_graph, 'knowledge_graph')
		print('Graph size:', len(edu_graph))
		print('Grammatical Clauses:', len(list(filter(lambda x: '{obj}' in x[1], edu_graph))))
		kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, edu_graph)
	del edu_graph

	if with_qa_dict_list: del qa_dict_list
	qa = AnswerRetriever(
		kg_manager, 
		CONCEPT_CLASSIFIER_OPTIONS, 
		SENTENCE_CLASSIFIER_OPTIONS, 
	)

	qa.load_cache(qa_cache)
	return qa, qa_cache

if __name__=='__main__':
	########################################################################
	### Extracting Knowledge Graph
	graph_cache = os_path.join(cache_path,f"graph_clauses_lemma-{GRAPH_BUILDER_OPTIONS['lemmatize_label']}_avoidjumps-{AVOID_JUMPS}.pkl")
	graph = load_or_create_cache(
		graph_cache,
		lambda: KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_documents_path(explainable_information_path, **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS)
	)
	save_graphml(graph, os.path.join(cache_path,'information_graph'))
	########################################################################
	### Get information to assess using the checklist. We evaluate the explainability of the answers to the questions of the checklist.
	qa, qa_cache = init_qa(graph, model_family='sbert', information_units='edu_amr_clause')
	# print(json.dumps(qa.gpt_cache, indent=4))

	with open(open_question_path if open_question_path else checklist_path,'r') as f:
		question_list = list(map(lambda x: x.strip(), f.readlines()))
	print(json.dumps(question_list, indent=4))
	question_answer_dict = {}
	for question in question_list:
		question_answer_dict.update(qa.ask([question], **OQA_OPTIONS))
	print(json.dumps(question_answer_dict, indent=4))
	# qa.summarise_question_answer_dict(question_answer_dict)

	### Define archetypal questions
	question_template_list = [
		##### AMR
		'What is {X}?',
		'Who is {X}?',
		'How is {X}?',
		'Where is {X}?',
		'When is {X}?',
		'Which {X}?',
		'Whose {X}?',
		'Why {X}?',
		##### Discourse Relations
		'In what manner is {X}?', # (25\%),
		'What is the reason for {X}?', # (19\%),
		'What is the result of {X}?', # (16\%),
		'What is an example of {X}?', # (11\%),
		'After what is {X}?', # (7\%),
		'While what is {X}?', # (6\%),
		'In what case is {X}?', # (3),
		'Despite what is {X}?', # (3\%),
		'What is contrasted with {X}?', # (2\%),
		'Before what is {X}?', # (2\%),
		'Since when is {X}?', # (2\%),
		'What is similar to {X}?', # (1\%),
		'Until when is {X}?', # (1\%),
		'Instead of what is {X}?', # (1\%),
		'What is an alternative to {X}?', # ($\leq 1\%$),
		'Except when it is {X}?', # ($\leq 1\%$),
		'{X}, unless what?', # ($\leq 1\%$).
	]

	### Define a question generator
	question_generator = lambda question_template,concept_label: question_template.replace('{X}',concept_label)

	########################################################################
	### Build explicanda
	print('Build explicanda')
	if checklist_path:
		with open(checklist_path,'r') as f:
			old_question_list_length = len(question_list)
			question_list = list(map(lambda x: x.strip(), f.readlines()))
			assert len(question_list) == old_question_list_length, f"Checklist has wrong length: {len(question_list)} instead of {old_question_list_length}"
	stats_list = []
	for question, answer_list in zip(question_list,question_answer_dict.values()):
		print('<Question>', question)
		confidence_list = [answer['confidence'] for answer in answer_list]
		if not confidence_list:
			print('No answer was found.')
			stats_list.append(0)
			continue
		
		content_list = [answer['sentence'] for answer in answer_list]
		print(f'<Answers> {json.dumps(content_list, indent=4)}')

		content_str = '\n'.join(map(lambda x: f'{x[0]}. "{x[1]}"', enumerate(content_list)))
		final_answer = qa.instruct_model(PROMPT_TEMPLATE(question,content_str), temperature=TEMPERATURE, top_p=TOP_P)
		qa.store_cache(qa_cache)
		time.sleep(60) # This is only to avoid hitting the 10.000 tokens-per-minute limit of the OpenaAI APIs
		# valid_index_str = '[0]'
		print(f'<Final Answer> {final_answer}')
		valid_index_set = set((
			idx
			for _,paragraph_citation in re.findall(r'[([]?(paragraph)s? *([&and, \d]+)[)\]]?', final_answer, re.IGNORECASE)
			for idx in re.findall(r'\d+', paragraph_citation)
		))
		print(f'<Valid Indexes> {valid_index_set}')
		if not valid_index_set:
			print('No valid answer was found.')
			stats_list.append(0)
			continue
		if checklist_path and final_answer.startswith('No,'):
			stats_list.append(0)
			continue
		####
		valid_confidence_list = [confidence_list[i] for i in map(int,valid_index_set)]
		max_confidence = max(valid_confidence_list)
		print(f'<Confidence> max: {max_confidence}, sum: {sum(valid_confidence_list)}, len: {len(valid_confidence_list)}')
		####
		valid_content_list = [content_list[i] for i in map(int,valid_index_set)]
		explicandum_graph = KnowledgeGraphExtractor(GRAPH_BUILDER_OPTIONS).set_content_list(valid_content_list, **GRAPH_CLEANING_OPTIONS).build(**GRAPH_EXTRACTION_OPTIONS)
		explanandum_aspect_list = get_concept_description_dict(graph=explicandum_graph, label_predicate=HAS_LABEL_PREDICATE, valid_concept_filter_fn=lambda x: '{obj}' in x[1]).keys()
		explanandum_aspect_list = list(explanandum_aspect_list)
		print('Important explicandum aspects:', len(explanandum_aspect_list), json.dumps(explanandum_aspect_list, indent=4))

		### Estimate DoX
		dox_qa, _ = init_qa(
			explicandum_graph, 
			model_family='sbert', 
			information_units='clause'
		)
		dox_estimator = DoXEstimator(dox_qa)
		dox = dox_estimator.estimate(
			aspect_uri_iter=list(explanandum_aspect_list), 
			query_template_list=question_template_list, 
			question_generator=question_generator,
			**ARCHETYPE_FITNESS_OPTIONS, 
		)
		print(f'<DoX>', json.dumps(dox, indent=4))
		### Compute the average DoX
		# archetype_weight_dict = {
		# 	'why': 1,
		# 	'how': 0.9,
		# 	'what-for': 0.75,
		# 	'what': 0.75,
		# 	'what-if': 0.6,
		# 	'when': 0.5,
		# }
		average_dox = dox_estimator.get_weighted_degree_of_explainability(dox, archetype_weight_dict=None)
		print('<Average DoX>', average_dox)
		compliance_score = average_dox*max_confidence
		print('<Compliance score>', compliance_score)
		stats_list.append(compliance_score)
	print('Average compliance score:', sum(stats_list)/len(stats_list))
	qa.store_cache(qa_cache)
