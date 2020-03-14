sudo docker run --gpus all --entrypoint /bin/bash -it -v /data/yijunq/MEVA_TSM_PROP:/MEVA_TSM_PROP -v /data/yijunq/MEVA_TSM_LABEL:/MEVA_TSM_LABEL -v /home/diva/temporal-shift-module:/temporal-shift-module -v /data/yijunq/models/pretrained:/pretrained -v /data/yijunq/models/checkpoints:/checkpoints tsm:v1

python main.py MEVA RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 32 -j 0 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --gpus 0 1 2 3 \
     --tune_from=/pretrained/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth


python test_models.py MEVA \
    --weights=/checkpoint/TSM_MEVA_RGB_resnet50_shift8_blockres_avg_segment8_e25/4.best.pth.tar \
    --test_segments=8 --test_crops=10 --topk 37\
    --batch_size=32  --actev --softmax \
    --gpus 0 1 2 3 --test_list /MEVA_TSM_LABEL/test_videofolder.txt


sudo docker run --gpus all --entrypoint /bin/bash -it -v /data/yijunq/kf1_test_TSM_prop:/MEVA_TSM_PROP -v /data/yijunq/MEVA_TSM_LABEL:/MEVA_TSM_LABEL -v /home/diva/temporal-shift-module:/temporal-shift-module -v /data/yijunq/models/pretrained:/pretrained  -v /data/yijunq/models/checkpoints:/checkpoint -v /data/yijunq/results:/results tsm:v2



=> Writing result to csv file: results.csv
test_models.py:333: RuntimeWarning: invalid value encountered in true_divide
  cls_acc = cls_hit / cls_cnt
[       nan 0.         0.27884615 0.         0.5245283  0.18947368
 0.46601942 0.01587302 0.         0.         0.59663866 0.43564356
 0.44525547 0.         0.65151515 0.31818182 0.                nan
 0.90632672 0.52525253 0.64761905 0.49122807 0.54098361 0.
 0.06206897 0.32743363 0.5        0.19565217 0.         0.05454545
 0.28846154 0.39130435 0.03225806 0.        ]



sudo docker run --rm --gpus all --entrypoint /bin/bash -it -v /data/diva/kf1_vod2sfix/kf1_tst/prop_gen/props:/props -v /data/diva/kf1_vod2sfix/kf1_tst/prop_gen/images/validation:/imgs -v /data/yijunq/MEVA_TSM_LABEL:/MEVA_TSM_LABEL -v /home/diva/temporal-shift-module:/temporal-shift-module -v /data/yijunq/models/pretrained:/pretrained  -v /data/yijunq/models/checkpoints:/checkpoint -v /data/yijunq/results:/results tsm:v2


python online_test.py MEVA \
    --weights=/checkpoints/TSM_MEVA_RGB_resnet101_shift8_blockres_avg_segment8_e25/8.best.pth.tar \
    --test_segments=8 --test_crops=10 --topk 37\
    --batch_size=32  --actev --softmax --full_res\
    --gpus 0 1 2 3 --test_list /MEVA_TSM_LABEL/test_videofolder.txt 


sudo docker run --rm --gpus all --entrypoint /bin/bash -it -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/props:/props -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/images:/imgs tsm:online_test


python online_test.py MEVA \
    --weights=/checkpoints/TSM_MEVA_RGB_resnet50_shift8_blockres_avg_segment8_e25_dense_nl/ckpt.best.pth.tar \
    --batch_size=24  --gpus 0 1 2 3  --test_crops 1 



sudo docker run --rm --gpus all -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/prop_gen/props:/props -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/prop_gen/images/validation:/imgs -v /mnt/ssda/results:/results actev_classification_tsm:v1 MEVA --weights=/checkpoints/TSM_MEVA_RGB_resnet50_shift8_blockres_avg_segment8_e25_dense_nl/ckpt.best.pth.tar --batch_size=12  --gpus 0 1 2  --dense_sample --test_crops 1

sudo docker run --rm --gpus all -v /data/yijunq/prop_gen/props:/props -v /data/yijunq/prop_gen/images/validation:/imgs -v /data/yijunq/results:/results actev_classification_tsm:v1 MEVA --weights=/checkpoints/TSM_MEVA_RGB_resnet50_shift8_blockres_avg_segment8_e25_dense_nl/ckpt.best.pth.tar --batch_size=16  --gpus 0 1 2 3 --test_crops 10


sudo docker commit --change='ENTRYPOINT ["python","/temporal-shift-module/online_test.py"]' 40115bf603f6 actev_classification_tsm:v3


sudo docker run -it --rm --gpus all  --entrypoint /bin/bash -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/props:/props -v /mnt/ssda/share/Cache/dec2020_kf1_iodvmz_1580983945/prop_kf1tst2536_iodvmz/images:/imgs -v /mnt/ssda/results:/results actev_classification_tsm:v1

sudo docker run -it --rm --gpus all --entrypoint /bin/bash  -v /data/yijunq/iod_kf1_tst_prop/prop_gen/props:/props -v /data/yijunq/iod_kf1_tst_prop/prop_gen/images/validation:/imgs -v /data/yijunq/results:/results actev_classification_tsm:v3


sudo docker run -it --rm --gpus all --entrypoint /bin/bash -v /home/diva/adaptive_temporal_shift_module:/adaptive_temporal_shift_module -v /data/yijunq/kinetics/imgs:/imgs -v /data/yijunq/kinetics400:/kinetics -v /data/yijunq/models/checkpoints:/checkpoints -v /data/yijunq/models/pretrained:/pretrained actev_classification_tsm:v2

