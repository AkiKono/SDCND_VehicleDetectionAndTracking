## Udacity SDCND Project 5: Vehicle Detection and Tracking
### Machine Learning and Computer Vision


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/VehicleImages.png
[image2]: ./output_images/NonVehicleImages.png
[image3]: ./output_images/hist32bin1.png
[image4]: ./output_images/hist32bin2.png
[image5]: ./output_images/hist32bin3.png
[image6]: ./output_images/hist16bin1.png
[image7]: ./output_images/hist16bin2.png
[image8]: ./output_images/hist16bin3.png
[image9]: ./output_images/SpacialBinColorSpace1.png
[image10]: ./output_images/SpacialBinColorSpace2.png
[image11]: ./output_images/SpacialBin1.png
[image12]: ./output_images/SpacialBin2.png
[image13]: ./output_images/SpacialBin3.png
[image14]: ./output_images/HOG1.png
[image15]: ./output_images/HOG2.png
[image16]: ./output_images/HOG3.png
[image17]: ./output_images/HOG4.png

[image20]: ./output_images/HOGHSV1.png
[image21]: ./output_images/HOGLUV1.png
[image22]: ./output_images/HOGHLS1.png
[image23]: ./output_images/HOGYUV1.png
[image24]: ./output_images/HOGYCrCb1.png
[image25]: ./output_images/HOGHSV2.png
[image26]: ./output_images/HOGLUV2.png
[image27]: ./output_images/HOGHLS2.png
[image28]: ./output_images/HOGYUV2.png
[image29]: ./output_images/HOGYCrCb2.png
[image30]: ./output_images/HOGHSV3.png
[image31]: ./output_images/HOGLUV3.png
[image32]: ./output_images/HOGHLS3.png
[image33]: ./output_images/HOGYUV3.png
[image34]: ./output_images/HOGYCrCb3.png
[image35]: ./output_images/HOGHSV4.png
[image36]: ./output_images/HOGLUV4.png
[image37]: ./output_images/HOGHLS4.png
[image38]: ./output_images/HOGYUV4.png
[image39]: ./output_images/HOGYCrCb4.png
[image40]: ./output_images/resultHSV.png
[image41]: ./output_images/resultLUV.png
[image42]: ./output_images/resultHLS.png
[image43]: ./output_images/resultYUV.png
[image44]: ./output_images/resultYCrCb.png
[image45]: ./output_images/report11.png
[image46]: ./output_images/report12.png
[image47]: ./output_images/report21.png
[image48]: ./output_images/report22.png

[image50]: ./output_images/window2.png
[image51]: ./output_images/window3.png
[image52]: ./output_images/window4.png
[image53]: ./output_images/window5.png
[image54]: ./output_images/window6.png
[image55]: ./output_images/heatmap1.png
[image56]: ./output_images/heatmap2.png
[image57]: ./output_images/heatmap3.png
[image58]: ./output_images/heatmap4.png
[image59]: ./output_images/heatmap5.png
[image60]: ./output_images/window6_.png
[image61]: ./output_images/window7.png
[image62]: ./output_images/window8.png
[image63]: ./output_images/window9.png
[image64]: ./output_images/window10.png


[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted Color and HOG features from the training images.

The code for this step is contained in the second through seventh code cells of the IPython notebook `vehicle_classifier.ipynb`.   

* 2nd code cell: Read in vehicle and non-vehicle images and show total # of images in each class
* 3rd code cell: Explore samples of images from vehicle and non-vehicle classes
* 4th code cell: Explore color histogram features for different color spaces
* 5th code cell: Explore spatial binning of different color spaces
* 6th code cell: Explore HOG features with different HOG parameters
* 7th code cell: Redefine feature extraction for training

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of the `vehicle` and `non-vehicle` images:

![alt text][image1]
![alt text][image2]

I then explored different color spaces with color histogram features with 16 bins and 32 bins for Vehicle and non-vehicle images. Here are examples of different color histogram with color spaces for 32 bins: RGB, HSV, LUV, HLS, YUV, YCrCb

![alt text][image3]
![alt text][image4]
![alt text][image5]

Here are examples of different color histogram with color spaces for 16 bins: RGB, HSV, LUV, HLS, YUV, YCrCb

![alt text][image6]
![alt text][image7]
![alt text][image8]

I chose to use 16 bins for the final training as it seems to have enough information to distinguish vehicle and non-vehicle images. As you can see from the figures above, RGB color space resulted in similar histograms for all of R, G, B channels. So RGB color space was eliminated from the list of color spaces.

I then explored other different color spaces with spatial binning color feature extraction. Here are examples of different spatial binning with color spaces with resized 8x8 image: HSV, LUV, HLS, YUV, YCrCb

![alt text][image9]
![alt text][image10]

YUV and YCrCb color space have quite similar spatial binning features.

Next, I explored HSV spatial binning with different image sizes: (8,8) (12,12) (20,20)

![alt text][image11]
![alt text][image12]
![alt text][image13]

In the final training, (8,8) image size was used. The reason will be explained in the SVM training section.

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here are examples of HOG features with different HOG parameters.

HOG parameters:  
`orientations` = 10  
`pixels_per_cell` = [8,10,12]  
`cells_per_block` = 2  

![alt text][image14]

HOG parameters:  
`orientations` = 8  
`pixels_per_cell` = [8,10,12]  
`cells_per_block` = 2  

![alt text][image15]

HOG parameters:  
`orientations` = 6  
`pixels_per_cell` = [8,10,12]  
`cells_per_block` = 2  

![alt text][image16]

HOG parameters:  
`orientations` = 6  
`pixels_per_cell` = [8,10,12]  
`cells_per_block` = 4  

![alt text][image17]

In the final training, the combination   
`orientations` = 8  
`pixels_per_cell` = 10  
`cells_per_block` = 2  
was used. The justification of this choice will be explained in the SVM training section.

The above examples were all using gray-scaled images.

Now, with all the HOG parameters fixed as below  
`orientations` = 8  
`pixels_per_cell` = 10  
`cells_per_block` = 2  
HOG images with different color space were explored for: HSV, LUV, HLS, YUV, YCrCb

Image 1: Vehicle

![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]

Image 2: Vehicle  

![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]

Image 3: Non-Vehicle  

![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]

Image 4: Non-Vehicle  

![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]
![alt text][image39]

It is not clear which color space seems to help classifying vehicles and non-vehicles.  
This will be discussed in the next section.

#### 2. Explain how you settled on your final choice of HOG parameters.

I trained classifier using LinearSVC with various combinations of color and HOG feature extraction parameters. First, I compared different color spaces. Below are the example results from LinearSVC training, comparing different color spaces: HSV, LUV, HLS, YUV, YCrCb

![alt text][image40]  

![alt text][image41]  

![alt text][image42]  

![alt text][image43]  

![alt text][image44]

I chose to use HSV because it resulted in the highest test accuracy of 99.32%.

Now, with HSV color space, I tried different parameters for feature extractions, and they were summarized in the table below.

|               |Trial1|Trial2|Trial3|Trial4|Trial5|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Test Accuracy  |99.21% |98.7%  |99.07% |99.01% |99.1%  |
|Time to Predict|0.00247|0.00241|0.00252|0.00243|0.00263|
|Time to Extract|72.91  |80.13  |67.32  |82.98  |82.52  |
|Orientations   |8      |6      |8      |8      |8      |
|Pixels/cell    |12     |10     |10     |10     |10     |
|Cells/block    |2      |2      |4      |2      |2      |
|Spatial Size   |(8,8)  |(8,8)  |(8,8)  |(6,6)  |(8,8)  ||
* Time to Predict: is a time to predict 10 labels
* Time to Extract: is a time to extract color and hog features

In the final training, the parameters for Trial5 was used.
(The reason why I did not use parameters used in Trial 1 was because I didn't realized Trial 1 was better than Trial 5 until when I started writing this report after finishing all tunings to complete vehicle detection on a project video!)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the eighth through ninth code cells of the IPython notebook `vehicle_classifier.ipynb`.

* 8th code cell: SVM Training with GridSearchCV  
* 9th code cell: Saving Vehicle Classifier

HOG feature, spatially binned color feature, and histogram of color feature were all put together as a one feature vector. The length of the final feature vector was 2640. Then, the values in the vector were normalized to zero mean and unit variance as shown below.

```python      
X = np.vstack((car_features, notcar_features)).astype(np.float64)               
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```

Then, feature vectors and labels were shuffled and separated into training set and test set in the ratio of 8:2.
```python
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

Next for the training, I used GridSearchCV to optimize SVM hyperparameters, Gamma and C for rbf kernel and C for linear kernel.

```python
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,1e-4],
                     'C': [1,10,50]},
                    {'kernel': ['linear'], 'C': [1,10,50]}]
```

Among various combination of parameters, the highest test accuracy achieved were 99.52% with {'gamma': 0.0001, 'kernel': 'rbf', 'C': 10}. It took 0.03662 Seconds to predict 10 labels with SVC. Below are the SVC report.

![alt text][image45]  
![alt text][image46]  

![alt text][image47]  
![alt text][image48]  



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the third and forth code cells of the IPython notebook `vehicle_detection_and_tracking.ipynb`.

I used HOG subsampling so that HOG features only needed to be extracted once. Overlap is set to two cells, so 20 pixels in the original image before scaling. Scaling is used to reduce processing time. Cars near front appears large with unnecessarily high pixel resolutions. So the searching area is resized as shown below.  
``` python
img_tosearch = img[ystart:ystop,:,:]
ctrans_tosearch = convert_color(img_tosearch, conv='HSV')
if scale != 1:
    imshape = ctrans_tosearch.shape
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
```
 Large window was used to search cars near front in the image, which was pre-scaled by large scale value. Whereas small window was used to search cars far front in the image, which was pre-scaled by small scale value to retain high enough pixel resolution for good search result. I tuned the window sizes and scales so the classifier can detect most of the cars in test images. Below is the table summarizing each windows size, scale, window search start y position, window search stop y position, and color of the window boxes shown in the test images.

|   |SIZE|SCALE|Y START|Y STOP|COLOR|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Window 1 |120x120|2|380|600|RED|
|Window 2 |60x60|1.5|380|600|BLUE|
|Window 3 |32x32|1|372|500|GREEN||

![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The examples of test images were shown above.
As a summary, I searched on three scales with three different window sizes using all HSV 3-channel HOG features plus spatially binned color with 8x8 spatial size plus histograms of color with 16 bins in the feature vector. To optimize the performance of the classifier, I tried many different combination of feature vector parameters, such as window size, scales, orientation, pixels_per_cell, cells_per_block for HOG feature, and spatial size for spatial binned color feature, and bin size for histograms of color feature. To optimize classifier results, I cross-validated different hyperparameters, such as rbf kernel with gamma =  [1e-3,1e-4] and C = [1,10,50] and linear kernel with C = [1,10,50]. The highest accuracy of 99.52% was achieved by the combination of rbf kernel with gamma = 1e-4 and C = 10. Even though the classifier was optimized, there are some false positives in the test images that needs to be taken care of. This will be a next topic in the following section.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the fifth and sixth code cells of the IPython notebook `vehicle_detection_and_tracking.ipynb`.

The process was explained step by step:

1. Heat map was created based on the positions of positive detections in each frame of the video.
2. Thresholded. When a vehicle was detected, often there are several positive detections overlaps whereas for false positive only one or two positives appear. So the each heat map was thresholded to eliminate unwanted false positives and leave high-confidence detections.
3. Labeled. The islands or blobs in the processed heatmap were considered to be vehicles with high-confidence. So each island or blob was labeled and numbered using `scipy.ndimage.measurements.label().`  
4. Visualized. To indicate the location of vehicles that are detected, rectangular boxes were created based on the location of each island or blob found in the previous step. Then the rectangular boxes were drawing in the original frame of the video and outputted.  

Below are the examples showing positive detections and the heatmaps created with threshold of 1.

Here are examples of positive detections and corresponding heatmaps and output images with rectangular boxes:

![alt text][image60] ![alt text][image55]
![alt text][image61] ![alt text][image56]
![alt text][image62] ![alt text][image57]
![alt text][image63] ![alt text][image58]
![alt text][image64] ![alt text][image59]

From the images above, you can see that the cars in the far front tend to have less overlapping of positive detections and thus have high chance of eliminated by thresholding. So the positive detection results from 32x32 windows were counted three times to prevent it from removed. The reason for choosing 32x32 windows was that, by looking at the images, most of the false positives were blue windows, and most of the green windows were true positives. Below are the actual implementation of heatmaps for reducing false positives in the project video processing.

1. Creat heatmap in each frame of the video.
2. Apply threshold of two
3. Append this heatmap in a deque defined in a class as a global variable.
 This deque variable holds ten most recent heatmaps.
4. Take average of heatmaps contained in this deque variable.
5. Apply another threshold of two on this averaged heatmaps.
6. Label.
7. Draw boxes.
8. Final output.

All false and true positive detections for each frame of video can be found in the center top of the output video. Thresholded and averaged heatmaps can be found in the right top corner of the output video.

Here's a [link to my video result](./output_project_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One shortcoming is that this vehicle detector cannot clearly identify how many cars are currently in the surrounding environment because cars that appear close each other will be detected as one car.

Another issue is that this implementation involved a lot of parameter tuning for this particular video of just about 50 seconds. Tuning these parameters and make the pipeline general enough to work on any driving environment can be difficult. For exmaple, this video was taken on freeway where only trees and mountains are shown in the frames, but if it was taken on a street with many different features, like buildings and stores with colors similar to cars, then my pipeline might fail.

Another case where my pipeline likely to fail is when there is a car driving at high relative speed. Such a car changes its position in the video frame quickly that the positive detections might not overlap enough and might be eliminated at thresholding process.

One option to make my pipeline more robust is to first increase the accuracy of classifier using more data and hard negative mining. If the classifier detects cars well enough, thresholding can be reduced.

Another improvement can be to use distance information to classify vehicles. If the detected car is located within the area of lane lines, then it is highly possible to be a vehicle. The distance can be obtained by LiDAR or roughly estimated by perspective transform and convert pixel dimension to actual distance.

References:   
[HOG](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
