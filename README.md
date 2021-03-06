# RoadlaneSegmentation
Conduct a road lane segmentation using VGG16 and tensorflow


### Introduction
This project deals with road lane semantic segmentation problem based on VGG16 and tensorflow. 
### Data
The data we are using is [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Then you can extract the dataset in `data` folder. A datafolder named `dataroad` will contain training set and test set.
### Implementation Details
#### Architecture
This is a `VGG16` model integrated with a fully-connected layer `FCN-8` as below shows.
![FCN-8](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/11/figure15.png)
<p align="center">
 <img src="./img/vgg16.png" alt="Overview" width="75%" height="75%">
 <br>VGG16
</p>

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
| epoch         | exec_time(s)  | training loss  |
| ------------- |:-------------:| --------------:|
| 1             | 17.67         | 6.2378         |
| 2             | 15.51         | 0.289          |
| 3             | 15.51         |   0.1813       |
| 4             | 17.67         | 6.2378         |
| 5             | 15.51         | 0.289          |
| 6             | 15.51         |   0.1813       |
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
