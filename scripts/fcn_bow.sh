cd ../
python trainers/fcn_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/fcn_bow_tfidf \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--epochs 10 \
--base_lr 2.5e-5

python trainers/fcn_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/fcn_bow_count \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--epochs 10 \
--base_lr 2.5e-5

python trainers/fcn_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/fcn_bow_binary_count \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--epochs 10 \
--base_lr 2.5e-5