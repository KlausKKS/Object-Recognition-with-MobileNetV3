The scripts have been built for a project to identify algae with microscope and camera (videostream). An appropriate dataset should be used for training (about 100 pictures per class the more the better).
The metrics for this approach are quite good: val_accuracy >93%, val_loss 0.2-0.3.
Included are features to zoom the videostream, measure the size of the object, save HD pictures and correct incorrect recognition.
These files can be used to train a MobileNetV3 model and create a mobilenet model_v3.keras. 
The file Objekterkennung_Zoom_mit_v3.py can de used for Object Recognition with opencv. A camera can be used for videostreaming. 
All runs on a regular PC.
