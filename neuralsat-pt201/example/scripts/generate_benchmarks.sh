python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_8_8_128_3_16_2
python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_8_8_128_3_32_2

python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_12_8_128_2_16_1
python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_12_8_128_3_16_2

python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_12_8_128_2_32_1
python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_12_8_128_3_32_2

python3 -m example.scripts.generate_instances --eps 0.005 --model_name vit_4_16_32_2_8_1 --model_type sigmoid

python3 -m example.scripts.generate_instances --eps 0.005 --model_name resnet20 --model_type resnet

python3 -m example.scripts.attack_benchmarks --eps 0.005 --model_name vit_4_16_32_2_8_1 --model_type sigmoid