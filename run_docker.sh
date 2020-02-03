sudo docker run --gpus all --entrypoint /bin/bash -it -v /media/hdd1/yijunq/MEVA_TSM_PROP:/MEVA_TSM_PROP -v /media/hdd1/yijunq/MEVA_TSM_LABEL:/MEVA_TSM_LABEL -v /home/divak/temporal-shift-module:/temporal-shift-module -v /media/hdd1/yijunq/checkpoint:/checkpoint -v /media/hdd1/yijunq/pretrained:/pretrained tsm:v1

python main.py MEVA RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 24 -j 0 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --gpus 0 1 2 3 --non_local --dense_sample\
     --tune_from=/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth