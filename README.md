The scripts have been built for a project to identify algae with microscope and camera (videostream). The scripts can be used to train and create a mobilenet model_v3.keras model and to perform Object-Recognition with OpenCV. The script rum on a MacBook (IOS) with M3 processor, for window computers adaptations may be required.

An appropriate dataset should be used for training (about 100 pictures per class (the more the better). Images should have the format 224x224 pixels. The training model delivers the following metrics: val_accuracy >95%, val_loss 0.2-0.3. At the end of the training a confusion matrix and a classification report are created.

Training should run with the following versions:
Python 3.10
tensorflow==2.16.1
numpy==1.26.4
pandas==2.2.2
Pillow>=10.0.0
scikit-learn>=1.4.0
opencv-python>=4.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
 
The file Objectrecognition.py can be used for Object Recognition with opencv. A usb camera can be used for videostreaming. Included are features to zoom the videostream, measure the size of the object, save HD pictures and correct incorrect recognition. To measure the size of the object microscope, camera und objective have to be calibrated in the script (pixel/µm}.

Training_mobilenet_v3_large.py runs through with the reference dataset (90 classes, ~8000 preprocessed images) without errors and produces mobilenet_model_v3_224.keras, confusion_matrix_224.png, and training_plot_v3.png.

Objectrecognition.py connects to the camera, displays live top-2 classifications, and optionally reaches confidence ≥ threshold for correctly detected objects.

Length measurement delivers plausible µm values with the active objective and correct pass/fail evaluation according to Measurement.csv 

A fresh clone with git clone https://github.com/KlausKKS/Object-Recognition-with-MobileNetV3.git
cd Object-Recognition-with-MobileNetV3 + pip install -r requirements.txt is sufficient to run both scripts.

