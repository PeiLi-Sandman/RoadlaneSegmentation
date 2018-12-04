# RoadlaneSegmentation
Conduct a road lane segmentation using VGG16 and tensorflow


### Introduction
This project deals with road lane semantic segmentation problem based on VGG16 and tensorflow. 
### Data
The data we are using is [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Then you can extract the dataset in `data` folder. A datafolder named `dataroad` will contain training set and test set.
### Implementation Details
#### Architecture
This is a `VGG16` model integrated with a fully-connected layer `FCN-8` as below shows.
![VGG16](<img src="./img/vgg16.png" alt="Overview" width="75%" height="75%">)
![FCN-8](https://www.researchgate.net/figure/Illustration-of-the-FCN-8s-network-architecture-as-proposed-in-20-In-our-method-the_fig1_305770331)

### Experiment and Result
 
This running results  through an Ubuntu 18.04
#### Result Examples
The good and bad results are shown below.
<p align="center">
 <img src="./img/example.png" alt="Overview" width="75%" height="75%">
 <br>Qualitative results.
</p>
<p align="center">
 <img src="./img/example1.png" alt="Overview" width="75%" height="75%">
 <br>Qualitative results.
</p>
<p align="center">
 <img src="./img/badexample.png" alt="Overview" width="75%" height="75%">
 <br>Not very good results.
</p>


#### Training Loss

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)


### Run

Run the following command to run the project:
```
python main.py
```
The trained model will be stored in `model` folder. And a csv file will be created to record the training loss.


**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.
