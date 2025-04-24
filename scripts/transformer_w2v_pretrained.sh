cd ../
python trainers/transformer_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transform_w2v_pretrained_frozen \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 64 \
--freeze_embeddings

python trainers/transformer_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--output_dir ./Results/transform_w2v_pretrained_learnable \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 64
