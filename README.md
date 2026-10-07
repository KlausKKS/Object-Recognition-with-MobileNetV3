The scripts have been built for a project to identify algae with microscope and camera (videostream). The scripts can be used to train and create a mobilenet model_v3.keras model and to perform Object-Recognition with OpenCV.

An appropriate dataset should be used for training (about 100 pictures per class (the more the better). Images should have the format 224x224 pixels. The training model delivers the following metrics: val_accuracy >95%, val_loss 0.2-0.3. At the end of the training a confusion matrix and a classification report are created.
 
The file Objekterkennung_Zoom_mit_v3.py can de used for Object Recognition with opencv. A camera can be used for videostreaming. Included are features to zoom the videostream, measure the size of the object, save HD pictures and correct incorrect recognition. To measure the size of the object microscope, camera und objective have to be calibrated (pixel/µm}.

