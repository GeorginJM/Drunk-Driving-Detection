import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from settings import image_folder_path, batch_size, epochs, initial_learning_rate, dropout_rate, l2_coefficient 

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Define the dropout rate
#dropout_rate = 0.5

# Define the L2 regularization coefficient
#l2_coefficient = 0.001

# Path to the folder containing your images
image_folder_path = r"images"

# Split the data into training and testing sets
image_files = []
labels = []

# Gather image files and labels from sober (0) and drunk (1) subdirectories
for label, subfolder in enumerate(['c0', 'c1']):
    folder_path = os.path.join(image_folder_path, subfolder)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                labels.append(label)

train_images, test_images, train_labels, test_labels = train_test_split(
    image_files, labels, test_size=0.2, random_state=42)

# Create an ImageDataGenerator instance for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=10,  # Random rotation up to 10 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    shear_range=0.2,  # Shear intensity
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

# Load and preprocess images using flow_from_directory for training
train_generator = datagen.flow_from_directory(
    image_folder_path,  # Path to the directory containing subdirectories of images
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=16,  # Batch size for training
    class_mode='binary',  # Use binary labels (0 or 1)
    shuffle=True  # Shuffle the data
)

# Load and preprocess images using flow_from_directory for testing
test_generator = datagen.flow_from_directory(
    image_folder_path,  # Path to the directory containing subdirectories of images
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,  # Batch size for testing
    class_mode='binary',  # Use binary labels (0 or 1)
    shuffle=False  # Do not shuffle the data for testing
)

# Define custom layers for feature detection
@register_keras_serializable(package='FlushedSkinDetection')
class FlushedSkinDetector(layers.Layer):
    def __init__(self):
        super(FlushedSkinDetector, self).__init__()

    def call(self, inputs):
        # Convert the image to the HSV color space
        hsv_image = tf.image.rgb_to_hsv(inputs)

        # Define lower and upper bounds for the red color range
        lower_red = tf.constant([0, 50, 50], dtype=tf.float32)
        upper_red = tf.constant([30, 255, 255], dtype=tf.float32)

        # Threshold the image to isolate red regions
        mask = tf.logical_and(tf.reduce_all(tf.greater_equal(hsv_image, lower_red), axis=-1),
                              tf.reduce_all(tf.less_equal(hsv_image, upper_red), axis=-1))

        # Calculate the ratio of flushed skin pixels to total pixels
        flushed_skin_ratio = tf.reduce_mean(tf.cast(mask, tf.float32), axis=[1, 2])

        return flushed_skin_ratio

@register_keras_serializable(package='BloodshotEyesDetection')
class BloodshotEyesDetector(layers.Layer):
    def __init__(self):
        super(BloodshotEyesDetector, self).__init__()

    def call(self, inputs):
        # Apply edge detection
        edges = tf.image.sobel_edges(tf.image.rgb_to_grayscale(inputs))
        edges_mag = tf.norm(edges, axis=-1)

        # Calculate the mean magnitude of edges
        mean_mag = tf.reduce_mean(edges_mag, axis=[1, 2])

        return mean_mag

@register_keras_serializable(package='PupilDilationDetection')
class PupilDilationDetector(layers.Layer):
    def __init__(self):
        super(PupilDilationDetector, self).__init__()

    def call(self, inputs):
        # Convert the image to grayscale
        gray_image = tf.image.rgb_to_grayscale(inputs)

        # Apply thresholding to segment the image (detecting the pupil)
        thresholded_image = tf.py_function(self.threshold_image, [gray_image], tf.float32)

        # Find contours in the thresholded image
        contours = tf.py_function(self.find_contours, [thresholded_image], tf.float32)

        # Assuming a threshold for pupil dilation
        dilation_threshold = 1000

        # Check if the number of contours (pupil detections) exceeds the dilation threshold
        pupil_dilation_detected = tf.cast(tf.shape(contours)[0] > dilation_threshold, tf.float32)

        # Reshape to have the desired output shape
        #pupil_dilation_detected = tf.reshape(pupil_dilation_detected, (-1,))

        # Reshape to have the desired output shape
        pupil_dilation_detected = tf.squeeze(pupil_dilation_detected)

        return pupil_dilation_detected

    def threshold_image(self, gray_image):
        # Ensure the grayscale image is in the correct format
        gray_image_uint8 = tf.image.convert_image_dtype(gray_image, tf.uint8)

        # Convert the grayscale image to numpy array
        gray_image_numpy = gray_image_uint8.numpy()

        # Ensure the image is in 8-bit unsigned integer format
        thresholded_image = cv2.convertScaleAbs(gray_image_numpy)

        # Apply thresholding to segment the image (detecting the pupil)
        _, thresholded_image = cv2.threshold(thresholded_image, 50, 255, cv2.THRESH_BINARY)

        # Convert the thresholded image to float32
        thresholded_image = tf.convert_to_tensor(thresholded_image, dtype=tf.float32)

        return thresholded_image


    def find_contours(self, thresholded_image):
        # Convert the thresholded image to numpy array and ensure it's in the correct format
        thresholded_image_numpy = thresholded_image.numpy().astype(np.uint8)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded_image_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flatten the list of contours and pad/truncate to a fixed length
        max_contours = 10  # Example maximum number of contours
        processed_contours = []
        for contour in contours:
            if len(contour) < max_contours:
                # Pad the contour with zeros to ensure a consistent length
                contour = np.concatenate((contour, np.zeros((max_contours - len(contour), 1, 2), dtype=np.float32)))
            elif len(contour) > max_contours:
                # Truncate the contour if it exceeds the maximum length
                contour = contour[:max_contours]
            processed_contours.append(contour)

        # Convert the processed contours to a TensorFlow tensor
        contours_tensor = tf.convert_to_tensor(processed_contours, dtype=tf.float32)

        return contours_tensor

@register_keras_serializable(package='SweatingDetection')
class SweatingDetector(layers.Layer):
    def __init__(self):
        super(SweatingDetector, self).__init__()

    def call(self, inputs):
        # Convert the input RGB image to uint8 format
        image_uint8 = tf.image.convert_image_dtype(inputs, tf.uint8)

        # Extract the RGB channels and convert them to float32
        r_channel = tf.cast(image_uint8[:, :, :, 0], tf.float32)
        g_channel = tf.cast(image_uint8[:, :, :, 1], tf.float32)
        b_channel = tf.cast(image_uint8[:, :, :, 2], tf.float32)

        # Calculate a sweating score based on color intensity
        sweating_score = (r_channel + g_channel + b_channel) / 3.0

        # Normalize the sweating score to the range [0, 1]
        sweating_score_normalized = sweating_score / 255.0

        return sweating_score_normalized

@register_keras_serializable(package='DroopyEyelidsDetection')
class DroopyEyelidsDetector(layers.Layer):
    def __init__(self):
        super(DroopyEyelidsDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

    def call(self, inputs):
        
        # Assuming the input is already in RGB format
        rgb_image = inputs

        # Define the EAR threshold and frame check
        thresh = 0.2

        # Load the face detector and facial landmark predictor        
        #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")      

        # Define a function to detect droopy eyelids
        def detect_droopy_eyelids(frame):
            # Convert the frame tensor to uint8
            frame_uint8 = tf.cast(frame, tf.uint8)
            # Convert frame_uint8 to numpy array
            frame_np = frame_uint8.numpy()
            # Detect faces using dlib
            subjects = self.detector(frame_np, 0)

            for subject in subjects:
                shape = self.predictor(frame, subject)  # Predict facial landmarks
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[36:42]  # Extract left eye coordinates
                right_eye = shape[42:48]  # Extract right eye coordinates
                left_ear = self.eye_aspect_ratio(left_eye)  # Compute left eye aspect ratio
                right_ear = self.eye_aspect_ratio(right_eye)  # Compute right eye aspect ratio
                ear = (left_ear + right_ear) / 2.0  # Compute average eye aspect ratio
                if ear < thresh:  # Check if eye aspect ratio is below threshold
                    return 1.0  # Return 1 indicating droopy eyelids detected
            return 0.0  # Return 0 indicating no droopy eyelids detected

        # Use tf.py_function to apply the function to each frame
        droopy_eyelids_detected = tf.map_fn(lambda frame: tf.py_function(detect_droopy_eyelids, [frame], tf.float32),
                                             rgb_image,
                                             dtype=tf.float32)

        # Reshape the output to match the expected shape
        return tf.reshape(droopy_eyelids_detected, (-1, 1))

    def eye_aspect_ratio(self, eye):
        # Compute EAR given the coordinates of the eye
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

# Define the input layer for images
image_input = layers.Input(shape=(224, 224, 3))

# Convolutional layers for image processing
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

# Dense (fully connected) layers for image processing
x = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_coefficient))(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_coefficient))(x)
x = layers.Dropout(dropout_rate)(x)

# Branch 1: Flushed skin detection
flushed_skin_output = layers.Reshape((1,))(FlushedSkinDetector()(image_input))

# Branch 2: Bloodshot eyes detection
bloodshot_eyes_output = BloodshotEyesDetector()(image_input)

# Branch 3: Pupil dilation detection
pupil_dilation_output = PupilDilationDetector()(image_input)
pupil_dilation_output = tf.expand_dims(pupil_dilation_output, axis=-1)  # Add a singleton dimension

# Branch 4: Sweating detection
sweating_output = SweatingDetector()(image_input)
sweating_output = tf.expand_dims(sweating_output, axis=-1)  # Add a singleton dimension for the channel
sweating_output = layers.GlobalAveragePooling2D()(sweating_output)

# Branch 5: Smile detection
# smile_output = SmileDetector()(image_input)

# Branch 6: Droopy eyelids detection
droopy_eyelids_output = layers.Reshape((1,))(DroopyEyelidsDetector()(image_input))

# Branch 7: Exaggerated frowns detection
# exaggerated_frowns_output = layers.Dense(1, activation='sigmoid', name='exaggerated_frowns')(x)

# Combine outputs with different weights
weighted_average = layers.Average()([
    flushed_skin_output,
    bloodshot_eyes_output,
    pupil_dilation_output,
    sweating_output * 0.4,  # Adjust the weight for sweating detection
    # smile_output * 0.7 , # Adjust the weight for smile detection
    droopy_eyelids_output 
])

# Create the model
model = models.Model(inputs=image_input, outputs=weighted_average)

# Define the learning rate schedule
#initial_learning_rate = 0.001

# Define the learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=3, monitor='loss', verbose=1)

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the data generator
#model.fit(train_generator, epochs, steps_per_epoch=len(train_generator), callbacks=[lr_scheduler])
model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_generator), callbacks=[lr_scheduler], batch_size=batch_size)

# Evaluate the model on the testing data
#test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator), batch_size=batch_size)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Display the model summary
model.summary()

model.save(r'model_latest_modified')
