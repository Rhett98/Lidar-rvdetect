# Lidar-inpainting

## train
python train.py -d ../dataset -ac config/salsanext_mos.yml
python train_dual.py -d ../dataset -ac config/dualrv.yml

## infer
python infer.py -d ../dataset -m data/model_salsanext_residual_1 -l log -s valid

## evaluate
python utils/evaluate_mos.py -d ../dataset -p log -s valid

## clean scan
python utils/scan_cleaner.py  ## modify config/post-processing.yaml

## viz scan(clean bin)
python utils/visualize.py -d data -s 8 -i -cl

## viz mos(bin & label)
python utils/visualize_mos.py -d../dataset -p log -s 8

## gen res image
python utils/auto_gen_residual_images.py

## simsiam train
python main_simsiam.py ../dataset -ac config/simsiam.yml
python main_simsiam.py ../toydata -ac config/simsiam.yml -dc config/labels/kitti-toy.yaml

python main_simsiam.py ../dataset -ac config/simsiam.yml -dc config/labels/semantic-kitti-mos.yaml

## fine-tuning
python train_finetuning.py -d ../dataset -ac config/salsanext_mos.yml -dc config/labels/semantic-kitti-mos.yaml -p checkpoint/checkpoint_0057.pth.tar