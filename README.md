# Working in Progress

We are updating our code. Please do not clone this repo yet.


# Fast and Accurate Online Video Object Segmentation via Tracking Parts

![Alt Text](https://github.com/JingchunCheng/FAVOS/blob/master/framework.png) 

Project webpage: <br />
Contact: Jingchun Cheng (chengjingchun14 at 163 dot com)

## Paper
[Fast and Accurate Online Video Object Segmentation via Tracking Parts]() <br />
Jingchun Cheng, Yi-Hsuan Tsai, Wei-Chih Hung, Shengjin Wang and Ming-Hsuan Yang <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

This is the authors' demo (single-GPU-version for Save) code for the DAVIS 2016 dataset as described in the above paper. Please cite our paper if you find it useful for your research.

```
@inproceedings{
}
```

## FAVOS Results
[Segmentation Comparisons with Fast Online Methods](https://www.dropbox.com/s/l95ozepuohie7x4/DAVIS16_segmentation_comparison_methods_with_strong_applicability.avi?dl=0)

[Example of Part Tracking](https://www.dropbox.com/s/3yszhdjz6klpmzr/Illustration_part_tracking.avi?dl=0)


## Requirements
* Install `caffe` and `pycaffe` (`opencv` is required). <br />
`cd caffe` <br />
`make all -j8` (paths are needed to change in the configuration file) <br />
`make pycaffe` <br />

* Download trained models and pre-computed results. <br />
Download [the DAVIS 2016 dataset](https://davischallenge.org/davis2016/code.html) and put it in folder "data" as "DAVIS2016". <br>
Download [segmentation results](https://www.dropbox.com/s/9zwob31bz91u75h/favos.tar?dl=0) and put them in folder "results". <br />
Download trained [ROISegNet model](https://www.dropbox.com/s/tkfa22j0ypq8ncq/ROISegNet_2016.caffemodel?dl=0) and put them in folder "models". <br />
Download [tracked parts](https://www.dropbox.com/s/pkqlzlhwun4qwuu/parts_DAVIS2016.tar?dl=0) and put them in foler "siamese-fc-master/tracking/". <br />

* Train your own ROISegNet. <br/>
Download [ResNet-101 model](https://github.com/KaimingHe/deep-residual-networks) and save it in floder "models" as "init.caffemodel" <br/>
`cd ROISegNet`<br/>
`python solve.py ../models/init.caffemodel solver_davis16.prototxt 0`<br/>

* Test our model. <br/>
We provide an example of testing script for our algorithm. <br/> 
`cd demo` <br/> 
`sh test_davis16_blackswan.sh` <br/> 
You can replace the class name "blackswan" in "test_davis16_blackswan.sh" with other classes in the DAVIS 2016 validation set to obatin results for other classes. <br/>


## Tracker
We use SiaFC tracker in "Fully-Convolutional Siamese Networks for Object Tracking". [here](https://github.com/bertinetto/siamese-fc) <br/>
Please download parts and tracking results from [here](https://www.dropbox.com/s/tkfa22j0ypq8ncq/ROISegNet_2016.caffemodel?dl=0). <br/>

## Download Our Segmentation Results on the DAVIS datasets
* FAVOS on DAVIS2016 [here](https://www.dropbox.com/s/9zwob31bz91u75h/favos.tar?dl=0)
* FAVOS on DAVIS2017 [here](https://www.dropbox.com/s/8gtcgf27qdhzyqu/favos_2017.tar?dl=0)


## Note
The model and code are available for non-commercial research purposes only.

