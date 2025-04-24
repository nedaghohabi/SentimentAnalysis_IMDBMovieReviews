cd ../
python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 100 \
--output_dir ./Results/fcn_w2v_trained_d100_frozen \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 300 \
--output_dir ./Results/fcn_w2v_trained_d300_frozen \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 500 \
--output_dir ./Results/fcn_w2v_trained_d500_frozen \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 100 \
--output_dir ./Results/fcn_w2v_trained_d100_learnable \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 300 \
--output_dir ./Results/fcn_w2v_trained_d300_learnable \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode train \
--vector_size 500 \
--output_dir ./Results/fcn_w2v_trained_d500_learnable \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--aggregation mean