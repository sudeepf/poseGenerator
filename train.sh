nice -n 10 python train.py \
--structure_string=64-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--data_split_string_train_2d=2D \
--batch_size=16 \
--joint_prob_max=1 \
--sigma=1.2 \
--gpu_string=0 \
--learning_rate=5e-6 \
--train_2d=false \
--dataset_dir=./Dataset/ \
--load_ckpt_path=./tensor_record//tmp/model64-64.ckpt

