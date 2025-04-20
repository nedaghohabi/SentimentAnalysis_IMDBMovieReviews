cd ../
python trainers/transformer_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--output_dir ./Results/lstm_w2v_frozen_onedir_no_attention \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--epochs 10