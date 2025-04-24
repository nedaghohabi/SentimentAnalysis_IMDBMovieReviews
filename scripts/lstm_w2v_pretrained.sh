cd ../
python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_pretrained_frozen_onedir_no_attention \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 128 \
--freeze_embeddings

python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_pretrained_frozen_bidir_no_attention \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 128 \
--freeze_embeddings \
--bidirectional


python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_pretrained_frozen_onedir_attention \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 128 \
--freeze_embeddings \
--use_attention


python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_pretrained_frozen_bidir_attention \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 128 \
--freeze_embeddings \
--bidirectional \
--use_attention

python trainers/lstm_word2vec.py \
--input_csv ./data/imdb_cleaned.csv \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_pretrained_learnable_bidir_attention \
--epochs 20 \
--base_lr 1e-4 \
--batch_size 128 \
--bidirectional \
--use_attention