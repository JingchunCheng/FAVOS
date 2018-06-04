# Working in Progress

We are updating our code. Please do not clone this repo yet.


# Fast and Accurate Online Video Object Segmentation via Tracking Parts (FAVOS)

![Alt Text](https://github.com/JingchunCheng/FAVOS/blob/master/figures/framework.png) 

Contact: Jingchun Cheng (chengjingchun14 at 163 dot com)

## Paper
[Fast and Accurate Online Video Object Segmentation via Tracking Parts]() <br />
Jingchun Cheng, Yi-Hsuan Tsai, Wei-Chih Hung, Shengjin Wang and Ming-Hsuan Yang <br />
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

This is the authors' demo (single-GPU-version) code for the DAVIS 2016 dataset as described in the above paper. Please cite our paper if you find it useful for your research.

```
@article{Cheng_favos_2018,
  author = {J. Cheng and Y.-H. Tsai and W.-C. Hung and S. Wang and M.-H. Yang},
  journal = {?},
  title = {Fast and Accurate Online Video Object Segmentation via Tracking Parts},
  year = {2018}
}
```

## FAVOS Results
[Segmentation Comparisons with Fast Online Methods](https://www.dropbox.com/s/l95ozepuohie7x4/DAVIS16_segmentation_comparison_methods_with_strong_applicability.avi?dl=0)

[Example Video of Part Tracking](https://www.dropbox.com/s/3yszhdjz6klpmzr/Illustration_part_tracking.avi?dl=0)


## Requirements
* caffe (pycaffe)
* opencv
* matlab
* A GPU with at least 12GB memory

## Download DAVIS 2016 dataset, trained models, tracked parts and pre-computed results. <br />
```
sh download_all.sh
```

## Test our model
We provide an example testing script `test_davis16.sh`. <br/>
```
# Please run download_all.sh first
# Usage: sh test_davis16.sh <GPU-id> <sequence-name>

sh test_davis16.sh 0 blackswan
```
The results would be saved in `results-demo/res_favos/<sequence-name>`.
You can replace the sequence name with others in the DAVIS 2016 validation set to obatin results for other videos. <br/>


## Train your own ROISegNet
Download [ResNet-101 model](https://www.dropbox.com/s/2506oyjkwy7acjv/init.caffemodel?dl=0) and save it in the folder "models" as "init.caffemodel" <br/>
```
cd ROISegNet
python solve.py ../models/init.caffemodel solver_davis16.prototxt 0
```

## Tracker
We use the SiaFC tracker in [Fully-Convolutional Siamese Networks for Object Tracking](https://github.com/bertinetto/siamese-fc). <br/>
The pre-computed parts and tracking results on DAVIS 2016 can be downloaded [here](https://www.dropbox.com/s/pkqlzlhwun4qwuu/parts_DAVIS2016.tar?dl=0). <br/>

Note that, we are currently working on a stable version to combine part tracking and ROISegNet for practical usage on any videos. We will update the code in a near future.

## Download Our Segmentation Results on the DAVIS datasets
* FAVOS on DAVIS2016 [link](https://www.dropbox.com/s/9zwob31bz91u75h/favos.tar?dl=0)
* FAVOS on DAVIS2017 [link](https://www.dropbox.com/s/8gtcgf27qdhzyqu/favos_2017.tar?dl=0)


## Note
The models and code are available for non-commercial research purposes only.

