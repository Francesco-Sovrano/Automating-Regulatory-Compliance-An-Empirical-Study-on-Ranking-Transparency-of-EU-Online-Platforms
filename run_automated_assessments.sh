source .env/bin/activate

cd expert_vs_gpt_vs_doxpy

################################
######## ChatGPT ###############
################################

cd code/gpt_based_approach/marketplaces/
python3 gpt3.5_assessment.py
python3 gpt4_assessment.py
cd ../../../

cd code/gpt_based_approach/search_engines/
python3 gpt3.5_assessment.py
python3 gpt4_assessment.py
cd ../../../

################################
######## DoXpert ###############
################################

cd code/doxpert/

mkdir logs
mkdir cache

CHECKLIST_PERTINENCE_THRESHOLD=0.3
DOX_ANSWER_PERTINENCE_THRESHOLD=0.3
SYNONYMITY_THRESHOLD=0.55

#### Amazon
mkdir cache/cache_exp4_amazon
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_marketplaces.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_marketplaces.txt \
	--explainable_information_path ../../data/platform_docs/marketplaces/amazon \
	--cache_path ./cache/cache_exp4_amazon \
	&> ./logs/exp4.amazon.fb.log.txt

#### Booking
mkdir cache/cache_exp4_booking
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_marketplaces.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_marketplaces.txt \
	--explainable_information_path ../../data/platform_docs/marketplaces/booking \
	--cache_path ./cache/cache_exp4_booking \
	&> ./logs/exp4.booking.fb.log.txt

#### Tripadvisor
mkdir cache/cache_exp4_tripadvisor
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_marketplaces.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_marketplaces.txt \
	--explainable_information_path ../../data/platform_docs/marketplaces/tripadvisor \
	--cache_path ./cache/cache_exp4_tripadvisor \
	&> ./logs/exp4.tripadvisor.fb.log.txt

#### Yahoo
mkdir cache/cache_exp4_yahoo
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_search_engines.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_search_engines.txt \
	--explainable_information_path ../../data/platform_docs/search_engines/yahoo \
	--cache_path ./cache/cache_exp4_yahoo \
	&> ./logs/exp4.yahoo.fb.log.txt

#### Bing
mkdir cache/cache_exp4_bing
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_search_engines.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_search_engines.txt \
	--explainable_information_path ../../data/platform_docs/search_engines/bing \
	--cache_path ./cache/cache_exp4_bing \
	&> ./logs/exp4.bing.fb.log.txt

#### Google
mkdir cache/cache_exp4_google
python3 doxpert_assessment.py \
	--model_type fb \
	--checklist_pertinence_threshold $CHECKLIST_PERTINENCE_THRESHOLD \
	--dox_answer_pertinence_threshold $DOX_ANSWER_PERTINENCE_THRESHOLD \
	--synonymity_threshold $SYNONYMITY_THRESHOLD \
	--checklist_path ../../data/checklist/checklist_for_search_engines.txt \
	--open_question_path ../../data/checklist/p2b_questions_for_search_engines.txt \
	--explainable_information_path ../../data/platform_docs/search_engines/google \
	--cache_path ./cache/cache_exp4_google \
	&> ./logs/exp4.google.fb.log.txt
