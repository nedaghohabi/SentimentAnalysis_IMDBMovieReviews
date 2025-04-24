cd ../
python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/fcn_w2v_pretrained_frozen \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings \
--aggregation mean

python trainers/fcn_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/fcn_w2v_pretrained_learnable \
--epochs 10 \
--base_lr 2.5e-5 \
--batch_size 128 \
--aggregation mean