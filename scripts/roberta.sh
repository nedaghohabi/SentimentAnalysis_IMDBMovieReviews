cd ../
python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/roberta_uncased_finetuned \
--logging_dir ./Results/roberta_uncased_finetuned/log \
--model_name "roberta-base" \
--tokenizer "roberta-base" \
--freeze_base

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/roberta_uncased_sst_finetuned \
--logging_dir ./Results/roberta_uncased_sst_finetuned/log \
--model_name "textattack/roberta-base-SST-2" \
--tokenizer "roberta-base" \
--freeze_base

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/roberta_uncased_trained \
--logging_dir ./Results/roberta_uncased_trained/log \
--model_name "roberta-base" \
--tokenizer "roberta-base"

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/roberta_uncased_sst_trained \
--logging_dir ./Results/roberta_uncased_sst_trained/log \
--model_name "textattack/roberta-base-SST-2" \
--tokenizer "roberta-base"