cd ../
python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model logreg \
--output_dir ./Results/logreg_tfidf

python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model logreg \
--output_dir ./Results/logreg_count

python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model logreg \
--output_dir ./Results/logreg_binary_count

python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model rf \
--output_dir ./Results/rf_tfidf

python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model rf \
--output_dir ./Results/rf_count

python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model rf \
--output_dir ./Results/rf_binary_count