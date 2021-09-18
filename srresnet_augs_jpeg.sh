CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_jpeg_50 --pretrain_epochs 20 --train_epochs 0 --jpeg_quality 50
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_jpeg_75 --pretrain_epochs 20 --train_epochs 0 --jpeg_quality 75
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_20_jpeg_100 --pretrain_epochs 20 --train_epochs 0 --jpeg_quality 100
CUDA_VISIBLE_DEVICES=4 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_jpeg_50 --pretrain_epochs 50 --train_epochs 0 --jpeg_quality 50
CUDA_VISIBLE_DEVICES=5 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_jpeg_75 --pretrain_epochs 50 --train_epochs 0 --jpeg_quality 75
CUDA_VISIBLE_DEVICES=6 TF_CPP_MIN_LOG_LEVEL=3 ./run_experiment.py --model srresnet_50_jpeg_100 --pretrain_epochs 50 --train_epochs 0 --jpeg_quality 100