DATA_ROOT=/n/fs/visualai-scr/Data/UCF101Images
python train.py \
--video_path=${DATA_ROOT}/ \
--annotation=${DATA_ROOT}/ucfTrainTestlist/ucf101_02.json \
--dropout_keep_prob=0.5 \
--num_scales=1 \
--learning_rate=1e-2 \
--batch_size=8 \

