safeNLP: OK
python3 neuralsat-pt201/main.py --net benchmarks/safeNLP/onnx/medical/perturbations_0.onnx --spec benchmarks/safeNLP/vnnlib/medical/hyperrectangle_54.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/safeNLP/onnx/ruarobot/perturbations_0.onnx --spec benchmarks/safeNLP/vnnlib/ruarobot/hyperrectangle_54.vnnlib --timeout 300

VNNComp23_NN4Sys: DOUBT
python3 neuralsat-pt201/main.py --net benchmarks/VNNComp23_NN4Sys/onnx/pensieve_big_parallel.onnx --spec benchmarks/VNNComp23_NN4Sys/vnnlib/pensieve_parallel_44.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/VNNComp23_NN4Sys/onnx/mscn_2048d.onnx --spec benchmarks/VNNComp23_NN4Sys/vnnlib/cardinality_0_4360_2048.vnnlib --timeout 300
INFO     01:46:59     [Failed] RandomAttack(seed=69, device=cpu)
Killed

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/nn4sys: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/nn4sys/onnx/pensieve_big_parallel.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/nn4sys/vnnlib/pensieve_parallel_54.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/nn4sys/onnx/lindex_deep.onnx --spec benchmarks/vnncomp2
023_benchmarks/benchmarks/nn4sys/vnnlib/lindex_7000.vnnlib --timeout 300

cora-vnncomp2024-benchmark: OK
python3 neuralsat-pt201/main.py --net benchmarks/cora-vnncomp2024-benchmark/nns/mnist-set.onnx --spec benchmarks/cora-vnncomp2024-benchmark/benchmark-files/mnist-img15.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/cora-vnncomp2024-benchmark/nns/svhn-trades.onnx --spec benchmarks/cora-vnncomp2024-benchmark/benchmark-files/svhn-img472.vnnlib --timeout 300

LinearizeNN_benchmark2024: OK
python3 neuralsat-pt201/main.py --net benchmarks/LinearizeNN_benchmark2024/onnx/AllInOne_50_50.onnx --spec benchmarks/LinearizeNN_benchmark2024/vnnlib/prop_50_50_0.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/LinearizeNN_benchmark2024/onnx/AllInOne_120_30.onnx --spec benchmarks/LinearizeNN_benchmark2024/vnnlib/prop_120_30_4.vnnlib --timeout 300


dist-shift-vnn-comp: OK
python3 neuralsat-pt201/main.py --net benchmarks/dist-shift-vnn-comp/onnx/mnist_concat.onnx --spec benchmarks/dist-shift-vnn-comp/vnnlib/index5924_delta0.13.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/dist-shift-vnn-comp/onnx/mnist_concat.onnx --spec benchmarks/dist-shift-vnn-comp/vnnlib/index4172_delta0.13.vnnlib --timeout 300

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/dist_shift: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/dist_shift/onnx/mnist_concat.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/dist_shift/vnnlib/index7227_delta0.13.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/dist_shift/onnx/mnist_concat.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/dist_shift/vnnlib/index1498_delta0.13.vnnlib --timeout 300

vnncomp2024_cifar100_benchmark: DOUBT
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2024_cifar100_benchmark/onnx/CIFAR100_resnet_medium.onnx --spec benchmarks/vnncomp2024_cifar100_benchmark/generated_vnnlib/CIFAR100_resnet_medium_prop_idx_1023_sidx_2206_eps_0.0039.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2024_cifar100_benchmark/onnx/CIFAR100_resnet_large.onnx --spec benchmarks/vnncomp2
024_cifar100_benchmark/generated_vnnlib/CIFAR100_resnet_large_prop_idx_132_sidx_6832_eps_0.0039.vnnlib --timeout 300
:
 INFO     02:17:10     [Failed] RandomAttack(seed=160, device=cpu)
Killed

vnncomp2024_tinyimagenet_benchmark: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2024_tinyimagenet_benchmark/onnx/TinyImageNet_resnet_medium.onnx --spec benchmarks/vnncomp2024_tinyimagenet_benchmark/generated_vnnlib/TinyImageNet_resnet_medium_prop_idx_3071_sidx_474_eps_0.0039.vnnlib --timeout 300:
INFO     02:25:49     [Failed] RandomAttack(seed=813, device=cpu)
Killed

python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2024_tinyimagenet_benchmark/onnx/TinyImageNet_resnet_medium.onnx --spec benchmarks/vnncomp2024_tinyimagenet_benchmark/generated_vnnlib/TinyImageNet_resnet_medium_prop_idx_667_sidx_2578_eps_0.0039.vnnlib --timeout 300
INFO     02:26:16     [Failed] RandomAttack(seed=276, device=cpu)
Killed

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/acasxu: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_2_7_batch_2000.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/acasxu/vnnlib/prop_1.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_4_4_batch_2000.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/acasxu/vnnlib/prop_3.vnnlib --timeout 300

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/cgan: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/cgan/onnx/cGAN_imgSz64_nCh_1.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/cgan/vnnlib/cGAN_imgSz64_nCh_1_prop_1_input_eps_0.005_output_eps_0.010.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/cgan/onnx/cGAN_imgSz32_nCh_1_transposedConvPadding_1.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/cgan/vnnlib/cGAN_imgSz32_nCh_1_transposedConvPadding_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib --timeout 300

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/collins_rul_cnn: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/collins_rul_cnn/onnx/NN_rul_full_window_40.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/collins_rul_cnn/vnnlib/monotonicity_CI_shift5_w40.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/collins_rul_cnn/onnx/NN_rul_small_window_20.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/collins_rul_cnn/vnnlib/monotonicity_CI_shift10_w20.vnnlib --timeout 300

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/metaroom: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/metaroom/onnx/4cnn_tz_57_9_no_custom_OP.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/metaroom/vnnlib/spec_idx_74_eps_0.00001000.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/metaroom/onnx/6cnn_ry_6_1_no_custom_OP.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/metaroom/vnnlib/spec_idx_137_eps_0.00000436.vnnlib --timeout 300

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/tllverifybench: OK
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/tllverifybench/onnx/tllBench_n=2_N=M=32_m=1_instance_3_0.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/tllverifybench/vnnlib/property_N=32_0.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023_benchmarks/benchmarks/tllverifybench/onnx/tllBench_n=2_N=M=64_m=1_instance_7_3.onnx --spec benchmarks/vnncomp2023_benchmarks/benchmarks/tllverifybench/vnnlib/property_N=64_3.vnnlib --timeout 300

ml4acopf_benchmark: BAD
python3 neuralsat-pt201/main.py --net benchmarks/ml4acopf_benchmark/onnx/300_ieee_ml4acopf.onnx --spec benchmarks/ml4acopf_benchmark/vnnlib/300_ieee_prop105.vnnlib --timeout 300: Infinite loop
Automatic inference of operator: sin
Automatic inference of operator: neg
Automatic inference of operator: cos
Automatic inference of operator: sin
Automatic inference of operator: neg
Automatic inference of operator: cos
Automatic inference of operator: sin
Automatic inference of operator: neg
Automatic inference of operator: cos
Automatic inference of operator: sin
Automatic inference of operator: neg
python3 neuralsat-pt201/main.py --net benchmarks/ml4acopf_benchmark/onnx/14_ieee_ml4acopf.onnx --spec benchmarks/ml4acopf_benchmark/vnnlib/14_ieee_prop12.vnnlib --timeout 300
Segmentation fault

vnncomp-benchmark-generation

vnncomp2023
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023/onnx/yolov5nano_LRelu_640.onnx --spec benchmarks/vnncomp2023/vnnlib/img_11752_perturbed_bbox_1_delta_0.005.vnnlib --timeout 300
python3 neuralsat-pt201/main.py --net benchmarks/vnncomp2023/onnx/yolov5nano_LRelu_640.onnx --spec benchmarks/vnncomp2023/vnnlib/img_8758_perturbed_bbox_0_delta_0.1.vnnlib --timeout 300

LSNC_VNNCOMP2024: raise OnnxConversionError util.misc.error.OnnxConversionError
python3 neuralsat-pt201/main.py --net benchmarks/LSNC_VNNCOMP2024/onnx/quadrotor2d_state.onnx --spec benchmarks/LSNC_VNNCOMP2024/vnnlib/quadrotor2d_state_17.vnnlib --timeout 300:     
python3 neuralsat-pt201/main.py --net benchmarks/LSNC_VNNCOMP2024/onnx/quadrotor2d_output.onnx --spec benchmarks/LSNC_VNNCOMP2024/vnnlib/quadrotor2d_output_19.vnnlib --timeout 300

Yolo-Benchmark:

https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/yolo
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/cctsdb_yolo
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/collins_yolo_robustness
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/ml4acopf
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/traffic_signs_recognition
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/vggnet16
https://github.com/ChristopherBrix/vnncomp2023_benchmarks/tree/main/benchmarks/vit

