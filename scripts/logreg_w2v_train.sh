cd ../
python trainers/ml_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 100 \
--workers 4 \
--model logreg \
--output_dir ./Results/logreg_w2v_trained_d100 \
--w2v_epochs 5

python trainers/ml_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 300 \
--workers 4 \
--model logreg \
--output_dir ./Results/logreg_w2v_trained_d300 \
--w2v_epochs 5

python trainers/ml_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 500 \
--workers 4 \
--model logreg \
--output_dir ./Results/logreg_w2v_trained_d500 \
--w2v_epochs 5