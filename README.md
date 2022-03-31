#  Leaf-level Image classification for Agriculture Applications

Leaf-level Image classification for Agriculture Applications: Detect pest attacks in the soybean crop that reduce yield and agricultural production to inspect and make decisions quickly.

Agriculture is one of the main challenges facing the world in the coming years. This is important in underdeveloped countries, not only because it plays a key role in achieving their development and poverty reduction goals, but also because it is the agricultural sector in these countries that is expected to meet the growing food needs. of humanity.

The state of food security and nutrition in the world” carried out in 2019 by the FAO (Food and Agriculture Organization of the United Nations), UNICEF and the WHO (World Health Organization) shows that in 2018 there were 821 million people with chronic malnutrition in the world, compared to 811 million the previous year, which corresponds to approximately one in nine people in the world. The FAO suggests that agricultural production will need to increase by around 70% by 2050 in order to feed a growing population.

As one of the world’s major and fastest expanding crops, soybean contributes significantly to overall human nutrition in terms of both calorie and protein intake; the crop appears to be well placed to meet the fast growing demand for vegetable oil and animal feed in developing countries. The yields of agriculture must necessarily grow in the medium term to be able to face the increase in demand.

Consistent improvements in average yield levels and reductions in production costs have steadily improved the competitive position of soybeans among arable crops, but there are still different factors that do not allow us to reach the maximum levels of performance, and pests are one of them. Pests and diseases in crops, in addition to attacking the eradication of hunger, generate great economic consequences. It is estimated that they cause the loss of up to 40% of food crops globally, which leads to a monetary loss of more than USD 220 billion each year. It is crucial, then, the development and implementation of new strategies that allow minimizing the consequences caused by this problem. Early detection of diseases and pests is a key factor in eradicating or minimizing the damage that this may cause.

Let's take a second to imagine the future: The human-machine relationship will be essential to solve society's problems. Artificial vision will be one of the main tools in our daily lives. And to solve this previously raised problem, using Artificial Intelligence seems to be an adequate solution.
To tackle this project, I imagined a huge soybean farm. It would be very difficult to manually inspect plant leaves because it would be time consuming. We can develop a system capable of detecting the leaves that were attacked by pests from images taken on the farm to make early decisions. And that is the idea of this project, an AI inference system for leaf-level image classification for agricultural applications.

I decided to make an image classification model using the resnet50 architecture with the Tensorflow framework. Vitis-AI contains ready-to-use deep learning models from Tensorflow. I followed the [Alveo U250 TF2 Classification Tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/12-Alveo-U250-TF2-Classification) to guide my steps. AI model zoo provides optimized and retrainable AI models for faster deployment, performance acceleration, and production.. AI model zoo provides optimized and retrainable AI models for faster deployment, performance acceleration, and production.
The workflow followed was as follows:

    Prepping the dataset

The project is based on the dataset of soybean leaves damaged by insects. Image capturing was done on several soybean farms, under realistic weather conditions, using two cell phones and a UAV. The data set consists of 3 categories: (a) healthy plants, (b) plants affected by caterpillars, and (c) images of plants damaged by Diabrotica speciosa with a total of 6410 images.Data Augmentation has been applied to them and they are standardized to a size of 224 x 224.

Since TensorFlow cannot use these folders directly as part of the training process, the data must be processed in a recognizable format which is the TF record. For it, the script is located on ["tf_record.py"](https://github.com/jrobador/resnet50-classification/blob/main/scripts/tf_records/tf_record.py).  This script is taken from [Imagenet to gcs](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py).

The TF record is a format used by the TensorFlow training environment to send data/labels to the model during training and validation. Small datasets can take advantage of other mechanisms for feeding data, but for larger datasets that don't fit in the available GPU memory, TF registers are the de facto mechanism for training. For it, the training fragment and validation fragment numbers were taken into account. TF's recommendation is that these shards be approximately 100MB each based on the following performance guide.

For validation images, raw images are needed for use with Vitis AI (for quantification and evaluation). The quantification, validation and test procedures use the TF validation records for these processes. The scripts provided with the Vitis AI zoo ResNet50 model expect raw images from a directory along with a tag text file that lists the image names and corresponding class id on each line.  *** REVISAR ESTA PARTE

    Training the Model

To train the model, 3 Python codes were used:
a- [The code that performs the training and validation](https://github.com/jrobador/resnet50-classification/blob/main/com/dataset.py)

b- [The code in charge of extracting characteristics from the dataset and providing functions necessary for training and validation](https://github.com/jrobador/resnet50-classification/blob/main/com/dataset.py)

c- [The code that describes functions used to process the images](https://github.com/jrobador/resnet50-classification/blob/main/com/images_preprocessing.py).

    Model Quantization and Compilation

Quantization: AI Quantizer is used to reduce model complexity without losing accuracy. The task of this is to convert weights and activations from 32-bit floating point to fixed point as INT8. The fixed-point network model requires less memory bandwidth, which provides faster speeds and higher power efficiency than the floating-point model. To run it on my model, I used the code described in model training by modifying the code flags. This generates a file with an .h5 type extension. 

Compilation: Maps the AI model to a high-efficient instruction set and data flow. Also performs optimizations such as layer fusion, instruction scheduling, and reuses on-chip memory as much as possible. For the compilation I used the script.sh located on the compile folder. This takes the quantized model, instantiates the .json file of the VCK5000 DPU (DPUCVDX8H), and generates the necessary output file to perform the deployment. This generates the .xmodel file, a meta.json and a text file md5sum.txt to verify the integrity. 


    Deployment on the VCK5000

Once the model has been compiled, a C++ API is needed for implementation. The Vitis AI development kit offers a unified set of high-level C++/Python programming APIs to facilitate the development of machine learning applications on Xilinx cloud-to-edge devices, with DPUCVDX8H for convolutional neural networks for VCK5000 devices. Provides the benefits of easily porting deployed DPU applications from the cloud to the edge or vice versa.

Before carrying out the deployment, it is necessary to install and verify some frequent problems that I encountered when I carried out the project.

Some software packages need to be installed for the board to work. You can follow the tutorial on the [VCK5000 Card Setup](https://github.com/Xilinx/Vitis-AI/tree/1.4.1/setup/vck5000).

  First, the ./install script found in Vitis-AI/setup/vck5000 must be run. This installs the Xilinx Runtime Library (XRT), Xilinx Resource Manager (XRT), and the V4E xclbin DPU for the vck5000.
  
  Second, some tools for flashing and validation must be installed.
  
In order to detect the board working, you must run:
 ``` sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan ```. After a restart this should show our VCK5000 is running version 4.4.6.
  ```
  ---------------------------------------------------------------------
Deprecation Warning:
    The given legacy sub-command and/or option has been deprecated
    to be obsoleted in the next release.
 
    Further information regarding the legacy deprecated sub-commands
    and options along with their mappings to the next generation
    sub-commands and options can be found on the Xilinx Runtime (XRT)
    documentation page:
    
    https://xilinx.github.io/XRT/master/html/xbtools_map.html

    Please update your scripts and tools to use the next generation
    sub-commands and options.
---------------------------------------------------------------------
Card [0000:03:00.0]
    Card type:		vck5000-es1
    Flash type:		OSPI_VERSAL
    Flashable partition running on FPGA:
        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]
    Flashable partitions installed in system:	
        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]

  ```
- ``` sudo /opt/xilinx/xrt/bin/xbmgmt flash --update  ```
 ```
---------------------------------------------------------------------
Deprecation Warning:
    The given legacy sub-command and/or option has been deprecated
    to be obsoleted in the next release.
 
    Further information regarding the legacy deprecated sub-commands
    and options along with their mappings to the next generation
    sub-commands and options can be found on the Xilinx Runtime (XRT)
    documentation page:
    
    https://xilinx.github.io/XRT/master/html/xbtools_map.html

    Please update your scripts and tools to use the next generation
    sub-commands and options.
---------------------------------------------------------------------
Card [0000:03:00.0]: 
	 Status: shell is up-to-date

Card(s) up-to-date and do not need to be flashed.
 ```
Finally, the validation utility can be used to verify that the board is working correctly:
 ``` /opt/xilinx/xrt/bin/xbutil validate --device 0000:03:00.1  ```
```
Starting validation for 1 devices

Validate Device           : [0000:03:00.1]
    Platform              : xilinx_vck5000-es1_gen3x16_base_2
    SC Version            : 4.4.6
    Platform ID           : 0x0
-------------------------------------------------------------------------------
Test 1 [0000:03:00.1]     : PCIE link 
    Warning(s)            : Link is active
                            Please make sure that the device is plugged into Gen 3x16,
                            instead of Gen 3x4. Lower performance maybe experienced.
    Test Status           : [PASSED WITH WARNINGS]
-------------------------------------------------------------------------------
Test 2 [0000:03:00.1]     : SC version 
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 3 [0000:03:00.1]     : Verify kernel 
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 4 [0000:03:00.1]     : DMA 
    Details               : Host -> PCIe -> FPGA write bandwidth = 3102.118468 MB/s
                            Host <- PCIe <- FPGA read bandwidth = 3294.490094 MB/s
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 5 [0000:03:00.1]     : iops 
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 6 [0000:03:00.1]     : Bandwidth kernel 
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 7 [0000:03:00.1]     : vcu 
Validation completed, but with warnings. Please run the command '--verbose' option for more details

Validation Summary
------------------
1  device(s) evaluated
1  device(s) validated successfully
0  device(s) had exceptions during validation

Validated successfully [1 device(s)]
  - [0000:03:00.1] : xilinx_vck5000-es1_gen3x16_base_2

Validation Exceptions [0 device(s)]

Warnings produced during test [1 device(s)] (Note: The given test successfully validated)
  - [0000:03:00.1] : xilinx_vck5000-es1_gen3x16_base_2 : Test(s): 'PCIE link'

```
Afterwards, the command source /workspace/setup/vck5000/setup.sh must be run to configure the work environment of the card.

Then, the following command must be used to avoid problems with the assignment of permissions:  ``` sudo chmod o=rw /dev/dri/render*  ```

To compile the software, we run  ``` ./build.sh ```. This script uses 3 files located in [src](https://github.com/jrobador/resnet50-classification/tree/main/deploy/src).

To run the model, run as follows: ``` .../resnet_bc .../VCK5000/resnet50_tf2_BC.xmodel .../path-to-images/ ```

The output classes delivered from this project model were as follows:
 ```
 Image : Diabroticaspeciosa_(128).jpg
top[0] prob = 0.982466  name = Diabrotica speciosa
top[1] prob = 0.010914  name = Caterpillar
top[2] prob = 0.006620  name = Healthy

Image : Caterpillar_(3181).jpg
top[0] prob = 0.877290  name = Caterpillar
top[1] prob = 0.122697  name = Diabrotica speciosa
top[2] prob = 0.000013  name = Healthy

Image : Diabroticaspeciosa_(64).jpg
top[0] prob = 0.878335  name = Diabrotica speciosa
top[1] prob = 0.118870  name = Healthy
top[2] prob = 0.002796  name = Caterpillar

Image : Healthy_(13).jpg
top[0] prob = 0.993477  name = Healthy
top[1] prob = 0.004060  name = Caterpillar
top[2] prob = 0.002463  name = Diabrotica speciosa

 ```

**Next Steps**

This project represents the first step applying Artificial Intelligence on Agriculture Applications. The results show the excellent efficiency that VCK5000 can provide. In the future I hope to create other models that can be joined with this one through a high-level API. Also, it is interesting to combine computer vision with machine learning models based on time series that allow farmers to predict the amount they are going to harvest based on the state of their crops' leaves. There are no solutions of this type on the market that offer a cloud-based service that is easy to use for the end consumer. 







