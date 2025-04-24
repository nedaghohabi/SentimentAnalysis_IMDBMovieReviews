cd ../

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=None" \
--output_dir ./Results/logreg_tfidf

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=None" \
--output_dir ./Results/logreg_count

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model logreg \
--model_args "penalty=None" \
--output_dir ./Results/logreg_binary_count



python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=l1,solver=liblinear,max_iter=10000" \
--output_dir ./Results/logreg_tfidf_l1

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=l1,solver=liblinear,max_iter=10000" \
--output_dir ./Results/logreg_count_l1

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model logreg \
--model_args "penalty=l1,solver=liblinear,max_iter=10000" \
--output_dir ./Results/logreg_binary_count_l1


python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=l2" \
--output_dir ./Results/logreg_tfidf_l2

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model logreg \
--model_args "penalty=l2" \
--output_dir ./Results/logreg_count_l2

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model logreg \
--model_args "penalty=l2" \
--output_dir ./Results/logreg_binary_count_l2



python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer tfidf \
--vectorizer_args "max_features=10000" \
--model rf \
--model_args "n_estimators=300,max_depth=40,min_samples_split=10,min_samples_leaf=2,max_features=sqrt,bootstrap=True,random_state=42" \
--output_dir ./Results/rf_tfidf

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000" \
--model rf \
--model_args "n_estimators=300,max_depth=40,min_samples_split=10,min_samples_leaf=2,max_features=sqrt,bootstrap=True,random_state=42" \
--output_dir ./Results/rf_count

python trainers/ml_bagofwords.py \
--input_csv ./data/imdb_cleaned.csv \
--vectorizer count \
--vectorizer_args "max_features=10000,binary=True" \
--model rf \
--model_args "n_estimators=300,max_depth=40,min_samples_split=10,min_samples_leaf=2,max_features=sqrt,bootstrap=True,random_state=42" \
--output_dir ./Results/rf_binary_count