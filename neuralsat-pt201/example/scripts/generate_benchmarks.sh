
python3 -m example.scripts.attack_benchmarks --eps 0.005 --model_name vit_4_16_32_2_8_1 --model_type sigmoid

python3 -m example.scripts.generate_instances_reconstruction --config train/config/cifar10_4.yaml --model_name cifar10_4 --seed 0 --device cpu --eps 0.001

python3 -m example.scripts.generate_instances_classification --model_type resnet --eps 0.0008 --model_name resnet12 --device cpu

grep -iRl 'unsat' example/generated_benchmark/resnet/eps_0.000400_resnet12/result/
grep -iRl 'unsat' example/generated_benchmark/resnet/eps_0.000500_resnet12/result/

python3 -m example.scripts.generate_instances_reconstruction --config train/config/cifar10_3.yaml --model_name cifar10_3 --seed 0 --device cpu --eps 0.001
python3 -m example.scripts.generate_instances_reconstruction --config train/config/cifar10_3_1.yaml --model_name cifar10_3_1 --seed 0 --device cpu --eps 0.001
python3 -m example.scripts.generate_instances_reconstruction --config train/config/cifar10_3_2.yaml --model_name cifar10_3_2 --seed 0 --device cpu --eps 0.001

python3 -m example.scripts.generate_instances_reconstruction --config train/config/cifar10_4_1.yaml --model_name cifar10_4_1 --seed 0 --device cpu --eps 0.001
 
python3 -m example.scripts.attack_benchmarks  --model_type resnet --eps 0.005 --model_name resnet_wide
python3 -m example.scripts.attack_benchmarks  --model_type resnet --eps 0.005 --model_name resnet_deep

python3 -m example.scripts.generate_instances_classification --model_type resnet --eps 0.005 --model_name resnet_wide --device cpu --seed 0 --simplify --epoch 4580
python3 -m example.scripts.generate_instances_classification --model_type resnet --eps 0.005 --model_name resnet_deep --device cpu --seed 0 --simplify --epoch 4801
python3 -m example.scripts.generate_instances_classification --model_type resnet --eps 0.005 --model_name resnet_base --device cpu --seed 0 --simplify

python3 -m example.scripts.generate_instances_classification --model_type resnet --eps 0.005 --model_name resnet_deep_2 --device cpu --seed 0 --simplify 
