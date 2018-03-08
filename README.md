## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, the goal is to write a software pipeline to identify the lane in a video.

The Project
-----------

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Instructions
------------

1. Read and follow the instructions in https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md to set up your development environment.

2. Clone this repository

> git clone https://github.com/vmonestel/CarND-Advanced-Lane-Lines.git

3. Open the code located in the IPYthon Notebook

The project code is in a jupyter notebook (Jupyter is an Ipython notebook where you can run blocks of code and see results interactively).

To start Jupyter in your browser, use terminal to navigate to the project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in point 1):

> jupyter notebook

5. A browser window will appear and show the contents of the current directory. Click on the file called "lane_finder.ipynb".

6. Another browser window will appear displaying the notebook. Hit "Run" in every box to run the code. The results are saved in the output folder.

Repo files
----------

* camera_cal: images for camera calibration
* test_images: images to test the solution
* output_images: sample images got from the pipeline stages
* camera_dist_pickle.p: file where the camera calibration parameters are stored
* writeup_template.md: includes a detailed explanation of the proposed solution
* project_video.mp4: original video where the lane is found
* project_video_lanes.mp4: output of project_video.mp4 after been processed
* lane_finder.ipynb: project code
