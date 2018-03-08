## Advanced Lane Finding Project

This project consists on the lane identification of images and videos. The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_corners.jpg "Chess board corners"
[image2]: ./output_images/undistorted_calibration.png "Undistorted chessboard"
[image3]: ./output_images/distorted_test_image.png "Distorted test image"
[image4]: ./output_images/undistorted_test_image.png "Undistorted test image"
[image5]: ./output_images/gradient_applied.png "Gradient applied"
[image6]: ./output_images/warped_img.png "Warped image"
[image7]: ./output_images/binary_warped_img.png "Binary Warped image"
[image8]: ./output_images/masked_image.png "Masked Warped image"
[image9]: ./output_images/histogram_example.png "Histogram"
[image10]: ./output_images/lanes.png "Lanes Identification"
[image11]: ./output_images/lanes_1.png "Lanes 1 Identification"
[image12]: ./output_images/final_img.png "Final image"
[video1]: ./project_video_lanes.mp4 "Video"

The code of the project is located in lane_finder.ipynb notebook.
### Camera Calibration

#### 1. Camera matrix and distortion coefficients.

The code for this step is contained in the "Finding Chessboard points" and the "Camera calibration" sections of the notebook.
The first step is to find the chessboard points. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the 3D world.  Here, I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_point` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is an example image after finding and drawing its corners:

![alt text][image1]

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The camera calibration parameters are saved in `camera_dist_pickle.p` file. Then, I applied this distortion correction to the chess image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Distortion-corrected image.

The camera calibration parameters are loaded from the file using `mtx_dist_load()`. Then, I applied the distortion correction to the test image using the `cv2.undistort()` function and I obtained this result (left image corresponds to the distorted image and right image corresponds to the undistorted image):

![alt text][image3]![alt text][image4]

#### 2. Thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps `color_thresh_apply()`).
I decided to use the HLS space, more specific I chose the L and S channels. The S channel identifies color intensity and helps to identify the yellow lines better, because the S values will be high for the bright colors. The L channel helps to identify better white lanes because they have high lightness values. The X gradient is used because it emphasizes edges closer to vertical like lane lines. Here's an example of my output for this step:

![alt text][image5]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `image_warp()`.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.array(
	[(190,imshape[0]),((imshape[1]/2) - 56,
	imshape[0] * 0.63),
	((imshape[1]/2) + 70, imshape[0] * 0.63),
	(imshape[1] - 145, imshape[0])], dtype=np.float32)
new_top_left = np.array([src[0,0],0])
new_top_right = np.array([src[3,0],0])
offset=[150,0]

dst = np.float32(
	[src[0] + offset,
	new_top_left + offset,
	new_top_right - offset,
	src[3] - offset])
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 340, 720      | 
| 584, 453.6    | 340, 0        |
| 710, 453.6    | 985, 0        |
| 1135, 720     | 985, 720      |

I verified that my perspective transform was working as expected and its warped counterpart was verified that the lines appear parallel in the warped image. The polygon image shows the src points on top of the undistorted image, and the warped image shows the eagle view of the lanes.

![alt text][image6]

The following image shows a binary image and its warped counterpart:

![alt text][image7]

Then, I decided to filter the warped binary images using a region of interest defined in `region_of_interest_mask()`. I used because it helps to improve the left and right lines finding. We are using the histogram information to get the 2 maximum peaks of pixels in the binary image; in some images there are some false lines identified near to the true lane lines and it makes harders to detect the lane. By aplying a region of interest, part of the fake lines are deleted from the image, as shown in the following image:

![alt text][image8]

#### 4. Lane-line pixels identification

The histogram provides great information about the lines location, we can have the amount of pixels for each x position of the image. I assume that after masking the images, the 2 highest peaks of the histogram represent both lines, as shown in the following image:

![alt text][image9]

The histogram information is split in he middle; the left peak is searched in the left half of the histogram and the righ peak is searched in the right half of the histogram. This allows to apply the windows slicing method to identify the lines. `find_lines()` implements the window slicing method. It starts with the histogram peaks and defines windows to search for valid line pixels. The next image shows in green the windows used, and in red/blue the lines.

![alt text][image10]

To fit the lanes, a 2nd order polynomial is used to get the coefficients.

![alt text][image11]

#### 5. Radius of curvature.

I did this in in `curvature_calculate()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`original_img_lines_draw()` gets the undistorted image, the warped image and the x points that represent the lines. It applies the inverse perspective transform using the warped_image, and draws a polygon on top of the undistorted image, which represents the lane of car. Also, it adds the curvature information of the lane.

![alt text][image12]

---

### Pipeline (video)

#### 1. Pipeline

The video processing is defined in the `pipeline()`. The pipeline function returns the fits, the `warped_image` and the curvatures of the lines. Then `draw_final_image()` takes the output from pipeline() to identify the lane in the original image.

Here's a [link to my video result](./project_video_lanes.mp4)

---

### Discussion

Initially, I faced some troubles to identy the lane correctly. In the video, in some cases the lanes were not parallel and specially the yellow line was not identified well. So I decided to improve my pipeline, first, I reduced the margin of the slicing windows from 110 to 50, it helped to identify the lines better. Another enhancement I implemented, was to use the previous image fit coefficients to find the lines in the current image. That helps to find more robust lines because it starts from the assumption that the current video image is very similar to the previous one, so the coefficients will be similar too. Also, the last enhancement reduced the amount of time to find new lanes, because it does not apply the slicing methods that starts from scratch.

The lane class is used to save some values got in the previous image, so if the current image curvature calculations are not in the valid range.

I tried my solution in the other harder videos, but it failed. For example, it detects a line in the middle of the lane because there is a fake line there. Also, it seems that the polygon used to warped the image is large, so it goes beyond the lane in the curves. My fist guess is that the thresholding method should be improved because it detecs some invalid lines and confuses the algorithm; another channels could be tried and/or another image space as well. The polygon can be reduced to see if it helps to find a better solution.
