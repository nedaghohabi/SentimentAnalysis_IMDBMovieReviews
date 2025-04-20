cd ../
python trainers/lstm_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_frozen_onedir_no_attention \
--epochs 5 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings False \
--bidirectional False \
--use_attention False

python trainers/lstm_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_frozen_bidir_no_attention \
--epochs 5 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings False \
--bidirectional True \
--use_attention False


python trainers/lstm_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_frozen_onedir_attention \
--epochs 5 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings False \
--bidirectional False \
--use_attention True


python trainers/lstm_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/lstm_w2v_frozen_bidir_attention \
--epochs 5 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings False \
--bidirectional True \
--use_attention True