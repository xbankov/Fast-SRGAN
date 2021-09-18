CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_10 --pretrain_epochs 1 --train_epochs 10
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_20 --pretrain_epochs 1 --train_epochs 20
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_50 --pretrain_epochs 1 --train_epochs 50
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_l3_10 --lr 0.001 --pretrain_epochs 1 --train_epochs 10
CUDA_VISIBLE_DEVICES=5 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_l3_20 --lr 0.001 --pretrain_epochs 1 --train_epochs 20
CUDA_VISIBLE_DEVICES=6 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srgan_l3_50 --lr 0.001 --pretrain_epochs 1 --train_epochs 50
