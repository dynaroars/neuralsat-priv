python3 -m scripts.train_classification --dataset torch/cifar10 --batch_size 128 --max_epoch 5000 --model vit --output_name vit_6_4_128
python3 -m scripts.train_recon --config config/imagenet.yaml --dataset imagenet
python3 -m scripts.train_recon --config config/cifar10.yaml --dataset cifar10