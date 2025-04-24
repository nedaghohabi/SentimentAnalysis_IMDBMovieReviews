cd ../
python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode none \
--vector_size 100 \
--output_dir ./Results/fcn_scrath_d100 \
--epochs 20 \
--base_lr 2.5e-5 \
--batch_size 128 \
--vocab_size 40000 \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode none \
--vector_size 300 \
--output_dir ./Results/fcn_scrath_d300 \
--epochs 20 \
--base_lr 2.5e-5 \
--batch_size 128 \
--vocab_size 40000 \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode none \
--vector_size 500 \
--output_dir ./Results/fcn_scrath_d500 \
--epochs 20 \
--base_lr 2.5e-5 \
--batch_size 128 \
--vocab_size 40000 \
--aggregation mean