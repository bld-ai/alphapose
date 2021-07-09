python scripts/demo_inference.py \
--cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml \
--checkpoint pretrained_models/fast_421_res152_256x192.pth \
--sp \
--video examples/vid/PettyDishonestIntermediateegret-mobile.mp4 \
--gpus 0 \
--format coco \
\
--detector tracker \
--detbatch 1 \
--showbox \
\
--save_video \
--vis_fast \
--outdir /mnt/d/Desktop/ \
\
--profile \
--debug \