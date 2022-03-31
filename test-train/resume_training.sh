
cd ../com/ 
python train_eval_h5.py --model train_dir_final/resnet50_model_156.h5 \
					      --eval_only=false \
					      --batch_size=2 \
					      --eval_images=false \
					      --createnewmodel=false \
  					      --eval_image_path=/workspace/resnet50-classification/bugdetection-classification/model_archives/validation/val_images \
					      --label_offset=1 \
					      --gpus 0,1 \
					      --eval_batch_size=96
