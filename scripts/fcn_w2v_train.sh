cd ../
python trainers/fcn_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 100 \
--output_dir ./Results/fcn_w2v_trained_d100 \
--epochs 5 \
--base_lr 2.5e-5

python trainers/fcn_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 300 \
--output_dir ./Results/fcn_w2v_trained_d300 \
--epochs 5 \
--base_lr 2.5e-5

python trainers/fcn_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode train \
--vector_size 500 \
--output_dir ./Results/fcn_w2v_trained_d500 \
--epochs 5 \
--base_lr 2.5e-5