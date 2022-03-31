import numpy as np
import cv2
import os
labels= ["Caterpillar", "Diabrotica speciosa", "Healthy"]

validation_images="/workspace/resnet50-classification/bugdetection-classification/val"
validation_output="/workspace/resnet50-classification/bugdetection-classification/model_archives/validation/val_images"
validation_labels="/workspace/resnet50-classification/bugdetection-classification/model_archives/validation/val_labels.txt"


validation_labels_file= open(validation_labels,"w")

index=1

for (dirpath, dirnames, filenames) in os.walk(validation_images):
    for filename in filenames:
        folder=dirpath.split(os.sep)
        folder = folder[-1]
        output_label_name = folder
        output_label_name = output_label_name.replace(" ","")
        label_idx = labels.index(folder)
        image = cv2.imread(dirpath+ '/'+ filename)
        output_filepath = validation_output + '/' + output_label_name + filename
        cv2.imwrite(output_filepath,image)
        validation_labels_file.write(output_label_name + filename + " " + str(label_idx) + "\n")
        print("wrote: ", output_filepath)
validation_labels_file.close()
