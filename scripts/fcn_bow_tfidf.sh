cd ../
python trainers/fcn_bagofwords.py \
--csv_dir ./data/IMDB\ Split \
--output_dir ./Results/fcb_bow_tfidf \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--epochs 5 \
--base_lr 2.5e-5