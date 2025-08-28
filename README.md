# Deepfake Detection (Meso 4)


## About the Project
This project implements the Meso-4 Convolution Neural Network (CNN) model for the purpose of deepfake detection.
The model was first introduced in 2019 in the paper "[MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888)".
Meso 4 was chosen because of its lightweight and efficient nature which allows it to be trained on a relatively small dataset of around 50K to 100k images and around 500 to 1,000 videos, unlike deeper networks like [Xception](https://arxiv.org/abs/1610.02357), and [EfficientNet](https://arxiv.org/abs/1905.11946) that need to be trained on atleast 100K to a million data points (both images and videos) for them to be able to generalize well enough.

#### Architecture of Meso 4:
* The model is made up of:
  * an input layer,
  * 4 convolutional layers (hence the name, Meso 4),
  * followed by a fully connected layer,
  * and finally an output layer.
* It has a total of 27,977 trainable parameters.
* The shape of the images passed to the network in the input layer is 256 * 256 * 3 (RGB).

#### Structure of the project:
* [models.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/models.py) contains implementations of the actual Meso-4 model for images and videos. It aslo contains the image and video data generators used to preprocess the dataset before the images and videos are passed to the neural network, as well as the program for training the models, so the training of the models is done by executing this script itself.
* [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) is the script that is going to make predictions. It accepts the path of an image, video, or a directory as its input and uses the trained models to predict whether the image/video is real or fake.
* [requirements.txt](https://github.com/Frame02496/DeepFake_Detection/blob/main/requirements.txt) contains the names of all the Python libaries that were used in this project, and are thus required to run it.
* [data](https://github.com/Frame02496/DeepFake_Detection/blob/main/data) is the dataset containing real and fake images that were used to train the image model.
* [video_data](https://github.com/Frame02496/DeepFake_Detection/blob/main/viideo_data) is the dataset containing real and fake videos that were used to train the video model.
* [saved_models](https://github.com/Frame02496/DeepFake_Detection/blob/main/saved_models) contains the image and video models trained on the previously mentioned datasets.
* [training_results](https://github.com/Frame02496/DeepFake_Detection/blob/main/training_results) contains the results of the training of the image and video models.
* [Example_data](https://github.com/Frame02496/DeepFake_Detection/blob/main/Example_data) is a small dataset containing real and fake images and videos that can be used to test the models.
## Steps to run the project

1. Download this repository as a zip or clone it to your local machine by running the following command:
``` bash
git clone https://github.com/Frame02496/DeepFake_Detection/
```
2. Use pip to install all dependencies listed in [requirements.txt](https://github.com/Frame02496/DeepFake_Detection/blob/main/requirements.txt):
``` bash
pip install -r requirements.txt
```
3. Run [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) and pass the path of an image, video, or a directory containing either two as an input to the script.
Example:
``` bash
python predictor.py /users/username/image.jpg
```
OR
``` bash
python predictor.py /users/username/video.mp4
```
OR
``` bash
python predictor.py /users/username/image_and_video_dir
```
[predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) will return a value between between 0 and 1 for each file.
How close a value is to 0 or 1 indicates how ceratin the models are about an image/video being real or fake.
The closer a value is to 0, the more ceratin the model is that the image/video is fake and vice versa.
If a value is close to 0.5, it indicates that the model is not entirely sure whether the corresponding image/video is real or fake.

4. Additionally, if a directory contains too many images, or videos and you want to limit the number of files that [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) should scan, you can use its options --limit, --ilimit, or --vlimt.
* --limit limits [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) from scanning more then the sprecified number of images and videos.
  Example:
  ``` python
  python predictor.py --limit 20 /users/username/image_and_video_dir
  ```
  When this command is executed only 20 images, and 20 videos will be scanned from the given directory.

* --ilimit limits [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) from scanning more then the sprecified number of images only.
  Example:
  ``` python
  python predictor.py --ilimit 20 /users/username/image_and_video_dir
  ```
  This makes it so that [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) only scan 20 images from the given directory but it does not affect the number of videos that are scanned; All videos that are present in the directory will be scanned.
  
* --vlimit limits [predictor.py](https://github.com/Frame02496/DeepFake_Detection/blob/main/predictor.py) from scanning more then the sprecified number of images only.
  Example:
  ``` python
  python predictor.py --ilimit 20 /users/username/image_and_video_dir
  ```
  When this command is executed only 20 images will be scanned from the given directory.
