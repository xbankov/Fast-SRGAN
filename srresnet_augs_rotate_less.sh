CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_rotate_0.5 --pretrain_epochs 20 --train_epochs 0 --rotate_angle 0.5
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_rotate_1 --pretrain_epochs 20 --train_epochs 0 --rotate_angle 1
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_rotate_2.5 --pretrain_epochs 20 --train_epochs 0 --rotate_angle 2.5
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_rotate_0.5 --pretrain_epochs 50 --train_epochs 0 --rotate_angle 0.5
CUDA_VISIBLE_DEVICES=5 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_rotate_1 --pretrain_epochs 50 --train_epochs 0 --rotate_angle 1
CUDA_VISIBLE_DEVICES=6 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_rotate_2.5 --pretrain_epochs 50 --train_epochs 0 --rotate_angle 2.5