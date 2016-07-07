# Learning to Assign Orientations to Feature Points

This software is a Python implemenation of the Learned Orientation Estimator presented in [1]. This software is intended to be used in conjuction with the [benchmark-orientation](https://github.com/kmyid/benchmark-orientation) repository. By default, the software does *not* use GPU, but can be easily enabled by configuring Theano to do so.

This software is strictly for academic purposes only.  For other purposes, please contact us.  When using this software, please cite [1] and other appropriate publications if necessary (see matlab/external/licenses for details).

[1] K.  M.  Yi, Y.  Verdie, P.  Fua, and V.  Lepetit.  "Learning to Assign Orientations to Feature Poitns.", Computer Vision and Patern Recognition (CVPR), 2016 IEEE Conference on.


Contact:

Kwang Moo Yi : kwang<dot>yi<at>epfl<dot>ch

Yannick Verdie : yannick<dot>verdie<at>epfl<dot>ch

## Requirements

* Theano
* Numpy
* OpenCV (2 or 3)

## Usage

In python-code folder

 ```python
 python runSingleTestWithFiles.py <image_file_name> <keypoint_file_name> <config_file_name> <output_file_name>
 ```

 - `image_file_name`: name of the image file do extract orientations.
 - `keypoint_file_name`: name of the keypoint file. In the form that [benchmark-orientation](https://github.com/kmyid/benchmark-orientation) repository uses.
 - `config_file_name`: configuration file for the model to test.
 - `output_file_name`: name of the output file. Will be in the same form as the `keypoint_file`
 
