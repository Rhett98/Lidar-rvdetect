# Lidar-inpainting

## train
python train.py -d /home/robot/Repository/data_odometry_velodyne/dataset -ac /home/robot/Repository/Lidar-inpainting/config/salsanext_mos.yml

## infer
python infer.py -d /home/robot/Repository/data_odometry_velodyne/dataset -m data/model_salsanext_residual_1 -l log -s valid

## evaluate
python utils/evaluate_mos.py -d /home/robot/Repository/data_odometry_velodyne/dataset -p log -s valid

## clean scan
python utils/scan_cleaner.py  ## modify config/post-processing.yaml

## viz scan(bin)
python utils/visualize.py -d data -s 8 -i -cl

## viz mos(bin & label)
python utils/visualize_mos.py -d /home/robot/Repository/data_odometry_velodyne/dataset -p log -s 8