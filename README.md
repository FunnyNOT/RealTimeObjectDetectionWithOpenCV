# RealTimeObjectDetectionWithOpenCV

The current project was implemented for a University project which cause was to understand how difficult can it be
to acquire useful data in modern machine learning problems. Its purpose is to try and sort through classification,
real-time images of fruits and nuts by using OpenCV. Inside the cv project folder can be found 5 .py files each with
a different purpose on our final program.After we apply a mask to the images so we can extract the features of the
valuable information, we analyze the images that we acquired and try to extract features like,hu_moments,
fd_haralick, and histograms. These features after being normalized (0,1), are split between train and test for our
models to be evaluated. After this is done, we build our pre-trained model based on the best model and we test it on new
data.

All the data acquired for this project was from a smartphone and OpenCV wasn't the best pick there would be.

The project could have been done with Keras easier and more efficiently but the whole purpose of it was to see why.


Things I might consider updating this code with:
Higher quality and quantity of data, more features for the classifiers to work with.


