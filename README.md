# resnet50-classification
Leaf-level Image classification for Agriculture Applications: Detect pest attacks in the soybean crop that reduce yield and agricultural production to inspect and make decisions quickly.

Agriculture is one of the main challenges facing the world in the coming years. This is important in underdeveloped countries, not only because it plays a key role in achieving their development and poverty reduction goals, but also because it is the agricultural sector in these countries that is expected to meet the growing food needs. of humanity.
The state of food security and nutrition in the world” carried out in 2019 by the FAO (Food and Agriculture Organization of the United Nations), UNICEF and the WHO (World Health Organization) shows that in 2018 there were 821 million people with chronic malnutrition in the world, compared to 811 million the previous year, which corresponds to approximately one in nine people in the world. The FAO suggests that agricultural production will need to increase by around 70% by 2050 in order to feed a growing population.
As one of the world’s major and fastest expanding crops, soybean contributes significantly to overall human nutrition in terms of both calorie and protein intake; the crop appears to be well placed to meet the fast growing demand for vegetable oil and animal feed in developing countries. The yields of agriculture must necessarily grow in the medium term to be able to face the increase in demand.
Consistent improvements in average yield levels and reductions in production costs have steadily improved the competitive position of soybeans among arable crops, but there are still different factors that do not allow us to reach the maximum levels of performance, and pests are one of them. Pests and diseases in crops, in addition to attacking the eradication of hunger, generate great economic consequences. It is estimated that they cause the loss of up to 40% of food crops globally, which leads to a monetary loss of more than USD 220 billion each year. It is crucial, then, the development and implementation of new strategies that allow minimizing the consequences caused by this problem. Early detection of diseases and pests is a key factor in eradicating or minimizing the damage that this may cause.
Let's take a second to imagine the future: The human-machine relationship will be essential to solve society's problems. Artificial vision will be one of the main tools in our daily lives. And to solve this previously raised problem, using Artificial Intelligence seems to be an adequate solution.
To tackle this project, I imagined a huge soybean farm. It would be very difficult to manually inspect plant leaves because it would be time consuming. We can develop a system capable of detecting the leaves that were attacked by pests from images taken on the farm to make early decisions. And that is the idea of this project, an AI inference system for leaf-level image classification for agricultural applications.
I decided to make an image classification model with the Tensorflow framework. Vitis-AI contains ready-to-use deep learning models from Tensorflow. AI model zoo provides optimized and retrainable AI models for faster deployment, performance acceleration, and production.
The workflow followed was as follows:

    Prepping the dataset

The project is based on the dataset of soybean leaves damaged by insects. Image capturing was done on several soybean farms, under realistic weather conditions, using two cell phones and a UAV. The data set consists of 3 categories: (a) healthy plants, (b) plants affected by caterpillars, and (c) images of plants damaged by Diabrotica speciosa with a total of 6410 images.Data Augmentation has been applied to them and they are standardized to a size of 224 x 224.
Since TensorFlow cannot use these folders directly as part of the training process, the data must be processed in a recognizable format which is the TF record. For it, the script is located on ["tf_record.py"](https://github.com/jrobador/resnet50-classification/blob/main/scripts/tf_records/tf_record.py).  This script is taken from [Imagenet to gcs](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py).
The TF record is a format used by the TensorFlow training environment to send data/labels to the model during training and validation. Small datasets can take advantage of other mechanisms for feeding data, but for larger datasets that don't fit in the available GPU memory, TF registers are the de facto mechanism for training. For it, the training fragment and validation fragment numbers were taken into account. TF's recommendation is that these shards be approximately 100MB each based on the following performance guide.
For validation images, raw images are needed for use with Vitis AI (for quantification and evaluation). The quantification, validation and test procedures use the TF validation records for these processes. The scripts provided with the Vitis AI zoo ResNet50 model expect raw images from a directory along with a tag text file that lists the image names and corresponding class id on each line.  *** REVISAR ESTA PARTE

    Training the Model

To train the model, 3 Python codes were used:
a- The code that performs the training and validation
b- The code in charge of extracting characteristics from the dataset and providing functions necessary for training and validation.
c- The code in charge of preprocessing the images if necessary.
[VER COMO SE ADJUNTA]

    Model Quantization and Compilation

Quantization: AI Quantizer is used to reduce model complexity without losing accuracy. The task of this is to convert weights and activations from 32-bit floating point to fixed point as INT8. The fixed-point network model requires less memory bandwidth, which provides faster speeds and higher power efficiency than the floating-point model. To run it on my model, I used the code described in model training by modifying the code flags. This generates a file of type [VER]
Compilación: Maps the AI model to a high-efficient instruction set and data flow. Also performs optimizations such as layer fusion, instruction scheduling, and reuses on-chip memory as much as possible. [EXPLICAR COMO LO HICE PARA MI VCK5000 Y EL ARCHIVO QUE SE GENERA]

    Deployment on the VCK5000

Once the model has been compiled, a C++ API is needed for implementation. The Vitis AI development kit offers a unified set of high-level C++/Python programming APIs to facilitate the development of machine learning applications on Xilinx cloud-to-edge devices, with DPUCVDX8H for convolutional neural networks for VCK5000 devices. Provides the benefits of easily porting deployed DPU applications from the cloud to the edge or vice versa. [ACA AGREGO LO QUE HICE CON LA COMPILACIÓN Y CON EL PROCESO DE HACER QUE FUNCIONE]

