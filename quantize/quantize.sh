cd ../com/
python train_eval_h5.py --model ./train_dir_final/resnet50_model_156.h5 \
	 				      --quantize=true \
					      --quantize_output_dir=/workspace/tf2_resnet50_imagenet_224_224_7.76G_1.4/code/com/vai_q_output \
					      --eval_only=true \
					      --eval_images=false \
					      --label_offset=1 \
					      --gpus=0
