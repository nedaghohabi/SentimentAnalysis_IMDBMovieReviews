cd ../
python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/distill_bert_uncased_finetuned \
--logging_dir ./Results/distill_bert_uncased_finetuned/log \
--model_name "distilbert-base-uncased" \
--tokenizer "distilbert-base-uncased" \
--freeze_base

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/distill_bert_uncased_sst_finetuned \
--logging_dir ./Results/distill_bert_uncased_sst_finetuned/log \
--model_name "distilbert-base-uncased-finetuned-sst-2-english" \
--tokenizer "distilbert-base-uncased" \
--freeze_base

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/distill_bert_uncased_trained \
--logging_dir ./Results/distill_bert_uncased_trained/log \
--model_name "distilbert-base-uncased" \
--tokenizer "distilbert-base-uncased"

python trainers/llm.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/distill_bert_uncased_sst_trained \
--logging_dir ./Results/distill_bert_uncased_sst_trained/log \
--model_name "distilbert-base-uncased-finetuned-sst-2-english" \
--tokenizer "distilbert-base-uncased"