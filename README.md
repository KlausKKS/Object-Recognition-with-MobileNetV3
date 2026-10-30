# Desmids Object Recognition

Object recognition using a trained MobileNetV3 model in TensorFlow Lite format. The original MobileNetV3 model which runs on an Apple M3 Chip was converted to the Lite format to allow running the application on Windows and Apple IOS.

The application uses a camera and OpenCV for real-time object recognition. It also supports displaying the two most likely classes, zooming, length measurements, and saving images.

## Requirements

- Windows 10 or Windows 11 (64-bit) or Apple IOS
- Python 3.11 (64-bit)
- Connected camera (settings in the script may have to be adapted)
- Files included in this repository

## Installation

Download or clone the repository:

```bash
git clone https://github.com/KlausKKS/Object-Recognition-with-mobilenetv3.git
cd Object-Recognition-with-mobileNetV3
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```
For IOS use python3 command, for Windows use python or py -3.11 command if python3 command does not work

## Running the Application

```bash
python3 objekterkennung_tflite_windows_mac.py
```
For IOS use python3 command, for Windows use python or py -3.11 command if python3 command does not work

## Model

The `mobilenet_model_v3_224.tflite` model is based on MobileNetV3Large and uses images with a resolution of 224 × 224 pixels. It runs on Mac and Windows.

The model has already been trained. No additional training is required to use the application.

## Controls

| Key | Function |
|---|---|
| + / - | Zoom in / out |
| 1–4 | Select objective magnification |
| s | Save image |
| c | Correct classification |
| a | Start image sequence capture |
| Space | Capture image in recording mode |
| r | Reset measurement |
| ESC | Exit current mode or close the application |

## Notes

Recognition accuracy depends on image quality, lighting conditions, the camera, and the classes used during training. Predictions should be verified before being used for scientific or professional purposes.


