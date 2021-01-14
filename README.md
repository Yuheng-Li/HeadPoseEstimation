# HeadPoseEstimation


This repository basically is a copy of https://github.com/natanielruiz/deep-head-pose, but I make some small modifications.<br />

The main change I made is adding image test script, which is test_on_image.py in code folder. (The original repo seems only support video as input) In order to run this script you need to specify two args: checkpoint path(please download it from the original repo) and image folder path. Please see code/test_on_image.py for details. <br />

Note that all images should be loosely cropped faces, so if you have a full body human images, then you can not use this model directly. <br />

The script will save a file called result.pkl which is a dict with four keys: 'file_name', 'yaw', 'pitch', 'roll'<br />
Something like this:<br />
{ <br />
'file_name':[ 'image1.png',  'image2.png', ....  ]<br />
'yaw':      [    12.56,          12.5,       ....  ]<br />
'pitch':    [    3.0,           -5.20,       ....  ]<br />
'roll':     [    1.02,           3.30,       ....  ]<br />
}<br />










