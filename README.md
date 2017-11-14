# Vehicle Detection

In this project I wrote a software pipeline that can process an image and identify vehicles in the image using a variety of Classifiers. 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The repository contains two main notebooks used for building the pipeline:

* playground.ipynb - used for feeling out different approaches and implementations. It has a number of visualisations of the techniques used in the next notebook. 
* Main.ipynb - The software pipeline used to process the video stream

[//]: # (Image References)
[image1]: ./readme_images/hog.png
[image2]: ./readme_images/sw1.png
[image3]: ./readme_images/sw2.png
[image4]: ./readme_images/sw3.png
[image5]: ./readme_images/sw4.png
[image6]: ./readme_images/hm1.png
[image7]: ./readme_images/hm2.png
[image8]: ./readme_images/hm3.png
[image9]: ./readme_images/hm4.png
[image10]: ./readme_images/hm5.png
[image11]: ./readme_images/hm6.png

## Feature Extraction

Features were extracted using the following techniques, 

* Histogram of Oriented Gradients 
* Spacial Binning of Color 
* Color Histogram

### Histogram of Oriented Gradients 

The function used to calculate the HOG is defined in the 4th code cell in the Main notebook and uses the sklearn hog function to calculate the Gradient orientations. 

In the playground notebook I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using 11 orientations with 16 pixels per cells and 2 cells per block. 

![alt text][image1]

I settled with this because it seemed to identify the most import features of the images with not much noise.

### Spacial Binning of Color

Since raw pixel values could still prove useful in identifying the vehicles. The image was scaled down to a size of 16px by 16px, at this stage it's not really noticable by the human eye but this pixel information acan still be used in making predictions. 

### Color Histogram

Here's another approach that utilizes the color distribution of cars to help in classification. The color space found most effective(through trial and failure) is the YCrCb color space. This approach uses numpy's `np.histogram` to identify the color features of cars with a bin rage of (0,256) and 16 bins. 

### Classification

Intially I trained a a Linear SVC classifier with high penalty C=50 but was not satisfied with the results it was achieving alone even with filtering methods applied later. To ensure as few false positives as early as possible two more classifiers were tuned and used in the classification process. These classifiers are, 

1. A decision tree with an entropy criterion and a min_samples_split of 20.
2. A Gaussian Naive Bayes Classifier

The code for the training process can be seen in the Main notebook code cells 11-13 under the Train the Classifier section. They achieved accuracies of 99%(Linear SVC), 96%(Decision Tree) and 95%(Gaussian Naive Bayes) respectively and the results of all three combined in the final classification process. The results of this approach did eliminate a number of false positives early on, but effectively did increase the number of false negatives(But it was workable :)). 

### Sliding Window search

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;): The function for the sliding widow search is defined in the 14th code cell and looks at window sizes of 64 within a focus area defined based on the position a car is more likely to appear. It only steps 1 cell at a time so it basically looks at every cell with no overlap and then combines the 3 classifiers to make a final decision on if this window contains a vehicle. Here's some sample images of the detections at a single scale,

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

The actual searching is done in the `find_boxes` function later down and searches at a number of scales to gather boxes. It searches the image at every scale from 1 to 3 with a .2 hop, so thats 10 scales in total, this worked well for me I think because with the use of 3 classifiers I'm fairly certain that the windows detected are positive detections. 

### Video implementation

There's a few Hiccups but here's a [link to my video result](./output_videos/project_video.mp4). There's also a few shorter videos in the folder that can be viewed. 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on them:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

### Discussion

Lot's of problems happened! Really had trouble getting one classifier to do a good job but I know it's possible and honestly I'm not sure I'm extracting the features as best as I could,there's definately more than enough room for improvement. I suspect if I had gathered data specifically from the videos being processed it would have done better but that wouldn't be a generalised approach.

To process videos faster and also using a little intuition after processing each frame of the video I skipped 10 frames using the same boxes found(I know it seems a little much but really in the average video can the car's position change that drastically). Note this was only done to speed up the processing of the video and to make the boxes a bit smoother throughout and I think it yielded 'ok' enough results. When I did process frame by frame there were a few noisy flashes so this approach also eliminated that completly. I'm excited to improve this pipeline and apply it to footage recorder here in Jamaica. I have a goal of bringing a Self Driving shuttle to the country and this pipeline combined with the Lane Detection pipeline can definately help me tune for roads and conditions here in the 3rd World.
