cd ../
python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--output_dir ./Results/lstm_w2v_trained_d100_bidir_attention \
--epochs 20 \
--vector_size 100 \
--base_lr 1e-4 \
--batch_size 128 \
--bidirectional \
--use_attention

python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--output_dir ./Results/lstm_w2v_trained_d300_bidir_attention \
--epochs 20 \
--vector_size 300 \
--base_lr 1e-4 \
--batch_size 128 \
--bidirectional \
--use_attention


python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--output_dir ./Results/lstm_w2v_trained_d500_bidir_attention \
--epochs 20 \
--vector_size 500 \
--base_lr 1e-4 \
--batch_size 128 \
--bidirectional \
--use_attention