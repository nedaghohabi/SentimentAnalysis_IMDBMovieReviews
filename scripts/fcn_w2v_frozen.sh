cd ../
python trainers/fcn_word2vec.py \
--csv_dir ./data/IMDB\ Split \
--w2v_mode pretrained \
--pretrained_path ./data/GoogleNews-vectors-negative300.bin \
--output_dir ./Results/fcn_w2v_frozen \
--epochs 5 \
--base_lr 2.5e-5 \
--batch_size 128 \
--freeze_embeddings False \
--aggregation mean