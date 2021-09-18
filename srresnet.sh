CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_10 --pretrain_epochs 10 --train_epochs 0
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20 --pretrain_epochs 20 --train_epochs 0
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50 --pretrain_epochs 50 --train_epochs 0
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_lr3_10 --lr 0.001 --pretrain_epochs 10 --train_epochs 0
CUDA_VISIBLE_DEVICES=5 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_lr3_20 --lr 0.001 --pretrain_epochs 20 --train_epochs 0
CUDA_VISIBLE_DEVICES=6 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_lr3_50 --lr 0.001 --pretrain_epochs 50 --train_epochs 0