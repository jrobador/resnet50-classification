vai_c_tensorflow2 -m /workspace/tf2_resnet50_imagenet_224_224_7.76G_1.4/code/com/vai_q_output/quantized.h5 \
		  -a /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json \
		  -o /workspace/tf2_resnet50_imagenet_224_224_7.76G_1.4/code/com/vai_q_output/VCK5000 \
		  -n resnet50_tf2_BC \
		  --options '{"input_shape": "1,224,224,3"}'
