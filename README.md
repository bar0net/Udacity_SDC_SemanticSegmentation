# Semantic Segmentation
### Introduction
The objective of this project is to build a Fully Connected Convolutional Neural Net to identify the road in a collection of pictures. The encoding of the FCN will be provided by a pre-trained VGG16 model and the decoder will be built using 1x1 convolutions, upscaling and layer skipping.

This project is part of Udacity's Self Driving Car Nanodegree Course. 

### Checklist
1. Ensure you've passed all the unit tests. [X]
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view). [X]
 
![Example1](/runs/1501984689.0793726/um_000032.png)

![Example2](/runs/1501984689.0793726/um_000090.png)
 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```

