from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from models import VideoDataGenerator
import numpy as np
import os
import argparse
import absl.logging

# ANSI Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
RESET ="\033[0m"

absl.logging.set_verbosity(absl.logging.ERROR)

# Create parsers
parser = argparse.ArgumentParser(description="Deepfake detection")
parser.add_argument('input_file', help='Path of an image, video, or directory')
parser.add_argument('--limit', type=float, default=float('inf') , help='Total number of images and videos to scan from a directory')
parser.add_argument('--ilimit', type=float, default=float('inf') , help='Total number of images to scan from a directory')
parser.add_argument('--vlimit', type=float, default=float('inf') , help='Total number of videos to scan from a directory')
args = parser.parse_args()
path = args.input_file

if os.path.exists(path):
    
    # If a file path is provided
    if os.path.isfile(path):
    
        # If provided path is an image
        if path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            model = load_model('saved_models/meso4_img.h5')
            for i in range(len(path)):
                img = load_img(path, target_size=(256, 256))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

            # Create the generator
            dataGenerator = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.2,
                horizontal_flip=True,
                rescale=1./255
            )

            # Create an iterator from the single image
            generator = dataGenerator.flow(img_array, batch_size=1)
            X = generator.__next__()
            pred = model.predict(X)[0][0]
            # Print name of image
            if path.rfind('/') != -1:
                print(f'{BLUE}{path[path.rfind('/')+1 : ]} {RESET}')
            else:
                print(f'{BLUE}{path[path.rfind('\\')+1 : ]} {RESET}')
            print(f'{CYAN}prediction: {pred:.4f} {RESET}')
            
            if pred >=0 and pred < 0.2:
                print(f'{RED}Highly certain that the image is fake\n {RESET}')
            elif pred >= 0.2 and pred <= 0.49:
                print(f'{RED}Fairly certain that the image is fake\n {RESET}')
            elif pred > 0.49 and pred < 0.51:
                print(f'{YELLOW}Unsure\n {RESET}')
            elif pred >= 0.51 and pred < 0.8:
                print(f'{GREEN}Fairly certain that the image is real\n {RESET}')
            else:
                print(f'{GREEN}Highly certain that the image is real\n {RESET}')
        
        # If provided path is a video
        elif path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            model_video = load_model('saved_models/meso4_vid.h5')
            video_generator = VideoDataGenerator(path, labels = [1])
            X, y = video_generator[0]
            pred = 1 - model_video.predict(X)[0][0]
            
            # Print name of video
            if path.rfind('/') != -1:
                print(f'{BLUE}{path[path.rfind('/')+1 : ]} {RESET}')
            else:
                print(f'{BLUE}{path[path.rfind('\\')+1 : ]} {RESET}')
            print(f'{CYAN}Prediction:', pred)
            
            if pred >=0 and pred < 0.2:
                print(f'{RED}Highly certain that the video is fake\n {RESET}')
            elif pred >= 0.2 and pred <= 0.49:
                print(f'{RED}Fairly certain that the video is fake\n{RESET}')
            elif pred > 0.49 and pred < 0.51:
                print(f'{YELLOW}Unsure\n {RESET}')
            elif pred >= 0.51 and pred < 0.8:
                print(f'{GREEN}Fairly certain that the video is real\n {RESET}')
            else:
                print(f'{GREEN}Highly certain that the video is real\n {RESET}')
        
        # If provided file is not a video or image
        else:
            raise ValueError(RED + 'Input must be of one of the following types: .jpg, .jpeg, .png, .avi, .mp4, .mov, .mkv' + RESET)
    
    # If a directory if provided
    elif os.path.isdir(path):
        if len(os.listdir(path)) != 0:
            # Loading models
            model_img = load_model('saved_models/meso4_img.h5')
            model_video = load_model('saved_models/meso4_vid.h5')
            dataGenerator = ImageDataGenerator(rescale=1./255)
            image_paths = [os.path.join(path, fname) for fname in os.listdir(path) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            image_names = []
            for i in image_paths:
                if i.replace(path+'/', '') != i:
                    image_names.append(i.replace(path+'/', ''))
                else:
                    image_names.append(i.replace(path+'\\', ''))

            video_paths = [os.path.join(path, fname) for fname in os.listdir(path) if fname.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
            video_names = []
            for i in video_paths:
                if i.replace(path+'/', '') != i:
                    video_names.append(i.replace(path+'/', ''))
                else:
                    video_names.append(i.replace(path+'\\', ''))

            # Load images into an array
            images = [img_to_array(load_img(path, target_size=(256, 256))) for path in image_paths]
            images = np.array(images)
            
            print(f'{YELLOW}{len(image_names)} image(s) and {len(video_names)} video(s) found {RESET}')
            if len(image_names) == 0 and len(video_names) == 0: quit()
            
            # Predicting for images
            if len(image_names) != 0:
                img_limit = args.limit if args.limit < args.ilimit else args.ilimit
                if img_limit != float('inf') and img_limit < len(image_names):
                    img_limit = int(img_limit)
                    print(f'\nPredicting for {img_limit} images:')
                else:
                    img_limit = len(image_names)
                    print('\nPredicting for images:')

                generator = dataGenerator.flow(images, batch_size=1, shuffle=False)
    
                for i, batch in enumerate(generator):
                    X = generator.__next__()
                    try:
                        print(f'{BLUE}{i+1}. {image_names[i]} {RESET}')
                    except IndexError:
                        break
                    pred = model_img.predict(X)[0][0]
                    print(f'{CYAN}Prediction: {pred:.4f} {RESET}')
                    if pred >=0 and pred < 0.2:
                        print(f'{RED}Highly certain that the image is fake\n {RESET}')
                    elif pred >= 0.2 and pred <= 0.49:
                        print(f'{RED}Fairly certain that the image is fake\n {RESET}')
                    elif pred > 0.49 and pred < 0.51:
                        print(f'{YELLOW}Unsure\n {RESET}')
                    elif pred >= 0.51 and pred < 0.8:
                        print(f'{GREEN}Fairly certain that the image is real\n {RESET}')
                    else:
                        print(f'{GREEN}Highly certain that the image is real\n {RESET}')
                        
                    if i == img_limit-1: break
                      
            # Predicting for videos
            if len(video_names) != 0:
                
                vid_limit = args.limit if args.limit < args.vlimit else args.vlimit
                if vid_limit != float('inf') and vid_limit < len(video_names):
                    vid_limit = int(vid_limit)
                    print(f'\nPredicting for {vid_limit} videos:')
                else:
                    vid_limit = len(video_names)
                    print('Predicting for videos:')
                
                labels = np.zeros(vid_limit)
                video_generator = VideoDataGenerator(video_paths, labels)
                for i, batch in enumerate(video_generator):
                    X, y = video_generator[i]
                    try:
                        print(f'{BLUE}{i+1}. {video_names[i]} {RESET}')
                    except IndexError:
                        break
                    pred = 1 - model_video.predict(X)[0][0]
                    print(f'{CYAN}Prediction: {pred:.4f} {RESET}')
                    
                    if pred >=0 and pred < 0.2:
                        print(f'{RED}Highly certain that the video is fake\n {RESET}')
                    elif pred >= 0.2 and pred <= 0.49:
                        print(f'{RED}Fairly certain that the video is fake\n{RESET}')
                    elif pred > 0.49 and pred < 0.51:
                        print(f'{YELLOW}Unsure\n {RESET}')
                    elif pred >= 0.51 and pred < 0.8:
                        print(f'{GREEN}Fairly certain that the video is real\n {RESET}')
                    else:
                        print(f'{GREEN}Highly certain that the video is real\n {RESET}')
                    if i == vid_limit-1: break
        
        # If provied directory is empty
        else:
            raise ValueError(f'{RED}{path} is empty' + RESET)
            
# If provied path does not exist
else:
    raise ValueError(f'{RED}{path} not exist' + RESET)

    

