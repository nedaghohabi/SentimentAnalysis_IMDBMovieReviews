cd ../
python trainers/transformer_wordvectors.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_scratch_d100 \
--w2v_mode none \
--epochs 20 \
--vector_size 100 \
--vocab_size 40000 \
--max_seq_len 2000 \
--base_lr 1e-4 \
--batch_size 64

python trainers/transformer_wordvectors.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_scratch_d300e \
--w2v_mode none \
--epochs 20 \
--vector_size 300 \
--vocab_size 40000 \
--max_seq_len 2000 \
--base_lr 1e-4 \
--batch_size 64

python trainers/transformer_wordvectors.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transformer_scratch_d500 \
--w2v_mode none \
--epochs 20 \
--vector_size 500 \
--vocab_size 40000 \
--max_seq_len 2000 \
--max_seq_len 2000 \
--base_lr 1e-4 \
--batch_size 64
