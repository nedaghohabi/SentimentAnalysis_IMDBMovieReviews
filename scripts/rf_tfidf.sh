cd ../
python trainers/ml_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model rf \
--output_dir ./Results/logreg_tfidf