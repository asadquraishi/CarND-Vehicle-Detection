**Vehicle Detection Project**

The goals / steps of this project are the following:

* Write a software pipeline to identify vehicles in a video from a front-facing camera on a car
* This will be accomplished by:
  * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
  * Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
  * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
  * Run your pipeline on a video stream (start with the test\_video.mp4 and later implement on full project\_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
  * Estimate a bounding box for vehicles detected.
  
[//]: # (Image References)
[image1]: ./output_images/8_test_images.png "Pipeline run on test images"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1\. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1\. Explain how (and identify where in your code) you extracted HOG features from the training images and how you settles on your final choice of HOG parameters.

The code for this step is contained in the`get_hog_features` fucntion in the `detection.py` file.

The signature of the function is as follows:

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

```

The parameters for orientation, pixels\_per\_cell, and cells\_per\_block were arrived at through experimentation.

Hog features are extracted in the `extract_features` function which is also found in the `detection.py` file.

It has the following signature:

```
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True)

```

This function calls the `get_hog_features` function. As such, it is where the color space is selected. While the function allows the use of a number of colour spaces, Originally,  `YCrCb` and `HLS` and `HSV` were chosen over `RGB`, and `YUV` as they performed the best as is seen below in the SVC accuracy of each:

Channel | RGB | HSV | LUV | YUV | HLS | YCrCb
--------|-----|-----|-----|-----|-----|------
0 | 0.945 | 0.995 | 0.935 | 0.925 | 0.965 | 0.955
1 | 0.925 | 0.925 | 0.975 | 0.975 | 0.945 | 0.965
2 | 0.925 | 0.935 | 0.945 | 0.915 | 0.875 | 0.925
ALL | 0.925 | 0.975 | 0.975 | 0.975 | 0.995 | 0.995

However in subsequent tests when dealing with real data and all features, `YCrCb` with 'ALL' channels performed the best in my classifier.

Furthermore, using the features for binned color features and color histogram features in addition to the hog features improved the accuracy of the classifier. Aside from hog, these two latter functions are, respectivley, `bin_spatial` and `color_hist`, also found in the `detection.py` module. They have the following signatures:

```python
def bin_spatial(img, size=(32, 32))
def color_hist(img, nbins=32)
```

The final parameters chosen for all three features are:
```python
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (24, 24)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
```

Each of these had parameters chosen through repeated testing on both the test images and on the video.

I also wrote a function that allowed me run the pipeline frame by frame to get a better idea where the pipeilne was challenged:

```python
for index in range(670,1100):
    vidcap = cv2.VideoCapture('project_video.mp4')
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, image = vidcap.read()
    new_image = process_image(image)
    cv2.imshow('img', new_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```
I was able to find the frames related to a position in the video using the following code:

```python
vidcap = cv2.VideoCapture('project_video.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,13000)
frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
print(frame)
326.0
```

#### 2\. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier, `LinearSVC`, was trained in the function `train_classifier` in the `detection.py` module in the following way:
* First I extacted the filenames for all of the train and test car and non-car images
* Then I perform feature engineering on cars and non cars by calling the `extract_features` function which is also in the `detection.py` module.
  * `extract_features` first converts the color of the image if necessary
  * It then obtains the spatial/color features by calling `bin_spatial` and appends them to an empty feature vector
  * It then obtains histogram features by calling `color_hist` and appends them to the above now non-empty feature vector
  * Finally it will create a HOG feature vector with one or all the color channels and append it to the above feature vector
  * Finally it returns these features to the `train_classifier` function
* I following this by stacking car and non-car features in a way usable by sklearn's `StandardScaler()` and performing a scalar transform on them to normalize them using `StandardScaler()`
* I then generate the labels and split the data and labels into a training and test set
* Finally I fit a LinearSVC support vector machine

### Sliding Window Search

#### 1\. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in the function `find_cars` in the `detection.py` module.

I used the code provided in class where we:
1. Take the full frame and convert it to a HOG feature set
2. We then scale the image with the scale parameter passed into the function
3. We split the image into channels
4. Define blocks and steps to take
5. Loop through the x and y steps (sliding window)
6. Take a window of the image and a) get the hog features for the window b)resize it to 64x64 pixels (same size as the train data)
7. Obtain bin_spatial and color_hist features
8. Combine the features in the same order as when training the classifier
9. Make a prediction
10. If the prediction is a car, draw a bounding box and add it to a list of bounding boxes

Some of the key parameters I chose (based on experimentation) are:
```python
scale = 1.5
cells_per_step = 1
```
By chosing the above values I was able to capture additional detail and then filter out the associated false positives with a higher threshold (3 in my case).

#### 2\. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Running the pipeline against the test images produced the following results (my test images were a mix of those provided for the project and those I extracted from the video as they were ones the pipeline struggled with):

![alt text][image1]

### Video Implementation

#### 1\. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_with_vehicle_detection_3.mp4)

#### 2\. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used two methods for dealing with false positives, both of which can be found in the `detect_vehicles.py` file:

1. Keeping a running sum of heatmaps for a certain sized window using the parameter `heatmaps_threshold = 8` which can be found in line 22 in the `detect_vehicles.py` file. This reinforces objects which appear frame after frame. Note that although I was planning to implement this, when looking for help I found Stewart DeSoto’s [post](https://carnd-forums.udacity.com/questions/38555139/how-to-integrate-heatmap-over-several-frames) and therefore used a modified version of his solution
2. Creating a heatmap and then thresholding it to get rid of false positives (lines 40-56 in `detect_vehicles.py`)

Note that the heatmap is obtained from the list of bounding boxes returned by the `find_cars` function.

---

### Discussion

#### 1\. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The pipeline still struggles with the following:

1. It loses the car once it gets a certain distance from the camera. I tried using multiple scales however this didn't solve the problem. I suspect I may need to add more scaled up images to the training images. I may also be able to use thinner horizontal slices than those I tried, using different scales for each.
2. It still finds some false positives. I think some of these can be removed by increasing the heatmap threshold and dealing with loss of the car detection by using a vehicles class (which I started) to keep a rolling average of the heatmap over subsequent frames.