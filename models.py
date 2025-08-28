import numpy as np
import cv2
import os
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU, Conv3D, MaxPooling3D, TimeDistributed, LSTM, GlobalAveragePooling3D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Dimensions for both images and videos
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}
video_dimensions = {'height': 256, 'width': 256, 'channels': 3, 'frames': 24}

class VideoDataGenerator(Sequence):

    def __init__(self, video_paths, labels, batch_size=1, target_size=(256, 256), max_frames=24, shuffle=False):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.video_paths) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_videos = [self.video_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        X, y = self._generate_data(batch_videos, batch_labels)
        return X, y
    
    def _generate_data(self, video_paths, labels):
        X = np.empty((self.batch_size, self.max_frames, *self.target_size, 3))
        y = np.array(labels)
        if type(self.video_paths) is not str:
            for i, video_path in enumerate(video_paths):
                frames = self._load_video(video_path)
                X[i] = frames
        else:
            frames = self._load_video(self.video_paths)
            X[0] = frames
            
        return X, y
    
    def _load_video(self, video_path):
        """Load video and extract frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        # Pad or truncate to max_frames
        while len(frames) < self.max_frames:
            frames.append(np.zeros((*self.target_size, 3)))
        
        return np.array(frames[:self.max_frames])

class Classifier:
    def __init__(self):
        self.model = None
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    '''def load(self, path):
        if os.path.exists(path):
            self.model.load_weights(path)'''
            
    def save(self, path):
        self.model.save(path)

class Meso4(Classifier, Model):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    
    def init_model(self): 
        x = Input(shape=(image_dimensions['height'],
                        image_dimensions['width'],
                        image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

class Meso4Video(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    
    def init_model(self):
        x = Input(shape=(video_dimensions['frames'],
                        video_dimensions['height'],
                        video_dimensions['width'],
                        video_dimensions['channels']))
        
        # 3D Convolutional layers to capture temporal features
        x1 = Conv3D(8, (3, 3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x1)
        
        x2 = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x2)
        
        x3 = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x3)
        
        x4 = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = GlobalAveragePooling3D()(x4)
        
        y = Dense(64)(x4)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

class UnifiedMeso4(Classifier):
    def __init__(self, mode='image', learning_rate=0.001):
        self.mode = mode
        if mode == 'image':
            self.model = Meso4(learning_rate).model
        elif mode == 'video_3d':
            self.model = Meso4Video(learning_rate).model
        else:
            raise ValueError("Mode must be 'image', or 'video_3d'")

def prepare_video_data(video_dir, batch_size=1, max_frames=24):
    """Prepare video data generator"""
    video_paths = []
    labels = []
    

    for class_name in ['real', 'fake']:
        class_dir = os.path.join(video_dir, class_name)
        if os.path.exists(class_dir):
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_paths.append(os.path.join(class_dir, video_file))
                    labels.append(1 if class_name == 'real' else 0)

    
    return VideoDataGenerator(video_paths, labels, batch_size=batch_size, max_frames=max_frames)

def train_image_model(data_dir, epochs=10, batch_size=32):
    """Train the image-based Meso4 model"""
    print("Setting up image-based training...")
    
    # Create model
    meso = UnifiedMeso4(mode='image', learning_rate=0.001)
    print(type(meso))
    
    # Load pretrained weights
    '''if weights_path and os.path.exists(weights_path):
        print(f"Loading pretrained weights from {weights_path}")
        meso.load(weights_path)'''
    
    # Prepare data generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Training
    history = meso.model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1
    )
    
    return meso, history

def train_video_model(video_dir, model_type='video_3d', epochs=10, batch_size=2):
    """Train video-based Meso4 model"""
    print(f"Setting up video-based training with {model_type}...")
    
    # Create model
    meso = UnifiedMeso4(mode=model_type, learning_rate=0.0001)  # Lower LR for video models
    
    # Prepare video data
    video_generator = prepare_video_data(video_dir, batch_size=batch_size)
    
    # Split data for training/validation
    train_size = int(0.8 * len(video_generator))
    
    training_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        train_loss = 0
        for i in range(train_size):
            X_batch, y_batch = video_generator[i]
            loss = meso.fit(X_batch, y_batch)
            train_loss += loss[0]
            
            if i % 10 == 0:
                print(f"  Batch {i}/{train_size}, Loss: {loss[0]:.4f}")
        
        training_losses.append(train_loss / train_size)
        
        # Validation
        val_loss = 0
        val_acc = 0
        val_batches = len(video_generator) - train_size
        
        for i in range(train_size, len(video_generator)):
            X_batch, y_batch = video_generator[i]
            loss = meso.get_accuracy(X_batch, y_batch)
            val_loss += loss[0]
            val_acc += loss[1]
        
        validation_losses.append(val_loss / val_batches)
        print(f"  Training Loss: {training_losses[-1]:.4f}")
        print(f"  Validation Loss: {validation_losses[-1]:.4f}")
        print(f"  Validation Accuracy: {val_acc / val_batches:.4f}")
    
    return meso, (training_losses, validation_losses)


# Evaluate model performance
def evaluate_model(model, test_generator, is_video=False):
    predictions = []
    true_labels = []
    
    if is_video:
        for i in range(len(test_generator)):
            X_batch, y_batch = test_generator[i]
            pred = model.predict(X_batch)
            predictions.extend(pred.flatten())
            true_labels.extend(y_batch)
    else:
        # For image generator
        test_generator.reset()
        pred = model.predict(test_generator)
        predictions = pred.flatten()
        true_labels = test_generator.classes
    
    # Convert predictions to binary
    binary_predictions = (np.array(predictions) > 0.5).astype(int)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(true_labels, binary_predictions,
                              target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, binary_predictions))
    
    return predictions, true_labels

# Training models:
if __name__ == "__main__":
    # Configuration
    # Configuration
    IMAGE_DATA_DIR = './data/'  # Directory with real/fake subdirectories
    VIDEO_DATA_DIR = './video_data/'  # Directory with real/fake video subdirectories
    EPOCHS = 10
    
    # Choose training mode
    training_mode = 'image'  # Options: 'image', 'video'
    
    if training_mode == 'image':
        print("Training image-based model...")
        model, history = train_image_model(
            IMAGE_DATA_DIR,
            epochs=EPOCHS
        )
        model.save('meos4.h5')
        
        
    elif training_mode == 'video':
        print("Training video-based model...")
        model, history = train_video_model(
            VIDEO_DATA_DIR,
            model_type='video_3d',
            epochs=EPOCHS
        )
        model.save('meso4_vid.h5')
    
    print("Training completed!")

