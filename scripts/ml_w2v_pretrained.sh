cd ../


python trainers/ml_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--model logreg \
--model_args "penalty=l2" \
--output_dir ./Results/logreg_l2_w2v_pretrained \

python trainers/ml_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--model rf \
--model_args "n_estimators=300,max_depth=40,min_samples_split=10,min_samples_leaf=2,max_features=sqrt,bootstrap=True,random_state=42" \
--output_dir ./Results/rf_w2v_pretrained \