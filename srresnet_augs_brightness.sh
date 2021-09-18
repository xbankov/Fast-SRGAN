CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_brightness_5 --pretrain_epochs 20 --train_epochs 0 --brightness_adjustment 5
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_brightness_10 --pretrain_epochs 20 --train_epochs 0 --brightness_adjustment 10
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_brightness_25 --pretrain_epochs 20 --train_epochs 0 --brightness_adjustment 25
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_brightness_5 --pretrain_epochs 50 --train_epochs 0 --brightness_adjustment 5
CUDA_VISIBLE_DEVICES=5 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_brightness_10 --pretrain_epochs 50 --train_epochs 0 --brightness_adjustment 10
CUDA_VISIBLE_DEVICES=6 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_brightness_25 --pretrain_epochs 50 --train_epochs 0 --brightness_adjustment 25

