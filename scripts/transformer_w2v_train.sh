cd ../
python trainers/transformer_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_w2v_trained_d100 \
--w2v_mode train \
--epochs 20 \
--vector_size 100 \
--base_lr 1e-4 \
--batch_size 64

python trainers/transformer_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_w2v_trained_d300e \
--w2v_mode train \
--epochs 20 \
--vector_size 300 \
--base_lr 1e-4 \
--batch_size 64

python trainers/transformer_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_w2v_trained_d500 \
--w2v_mode train \
--epochs 20 \
--vector_size 500 \
--base_lr 1e-4 \
--batch_size 64
