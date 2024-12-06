import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, BatchNormalization, ReLU,
                                     GlobalAveragePooling2D, Dense, Reshape, Multiply)
input = 'testdir copy 2/label1/cam1/0014_c1_127.png.png'
import cv2
import mediapipe as mp
import numpy as np
import os




mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


image_path = 'testdir copy 2/label1/cam2/0014_c2_132.png.png'
output_path = 'pose_image6.png'

import cv2
import mediapipe as mp
import numpy as np

def process_pose_image(image_path, output_path):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Failed to load image {image_path}")
                return
            
            image_height, image_width, _ = image.shape
            
            # Resize the image
            image_resized = cv2.resize(image, (int(image_width * 0.5), int(image_height * 0.5)))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Process the image to detect pose
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"Warning: No pose landmarks found for {image_path}")
                return
            
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                keypoints.append((x, y))
            
            # Create an empty image for drawing the skeleton
            skeleton_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            
            # Draw connections between keypoints
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                cv2.line(skeleton_image, start_point, end_point, (0, 255, 0), 2)
                cv2.circle(skeleton_image, start_point, 4, (0, 0, 255), -1)
                cv2.circle(skeleton_image, end_point, 4, (0, 0, 255), -1)
            
            # Save the skeleton image
            cv2.imwrite(output_path, skeleton_image)
            print(f'Saved skeleton pose at {output_path}')
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    print("Processing complete.")
process_pose_image("testdir copy 2/label1/cam2/0014_c2_132.png.png", "pose_images7.jpg")


pose = "pose_images7.jpg"


# Load the image
image = cv2.imread(input)

# Resize the image
image_resized = cv2.resize(image, (128, 256))

# Apply minimal sharpening kernel to maintain clarity but not over-enhance
sharpening_kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3, -0.5],
                              [0, -0.5, 0]])
image_sharpened = cv2.filter2D(image_resized, -1, sharpening_kernel)

# Apply fixed brightness adjustment
image_bright = tf.image.adjust_brightness(image_sharpened, delta=0.2)  # Slight brightness increase for better color clarity

# Apply moderate contrast and saturation adjustments for better color visibility
image_bright = tf.image.adjust_contrast(image_bright, contrast_factor=1.15)  # Moderate contrast increase
image_bright = tf.image.adjust_saturation(image_bright, saturation_factor=1.2)  # Slight saturation increase

# Convert to float32 and normalize the image
imagef = tf.cast(image_bright, dtype=tf.float32) / 255.0

# Add a batch dimension (expand dims)
image_tensor = tf.expand_dims(imagef, axis=0)

# Convert the TensorFlow tensor to a NumPy array for display
image_np = image_bright.numpy()
image_np = image_np.clip(0, 255).astype('uint8')
img = image_tensor

poseimage = cv2.imread(pose)
if poseimage is None:
    print(f"Error: Unable to read the pose image from {pose}")
    exit(1)  # Exit or handle the error as appropriate
else:
    poseimage_resized = cv2.resize(poseimage, (128, 256))




imagef = tf.cast(poseimage_resized, dtype=tf.float32)

poseimage_tensor = tf.convert_to_tensor(imagef, dtype=tf.float32)


poseimage_tensor = poseimage_tensor / 255.0


poseimage_tensor = tf.expand_dims(poseimage_tensor, axis=0)
pose = poseimage_tensor


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):

        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):

        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)

        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        return self.gamma * normalized + self.beta

class ResBlock(tf.keras.Model):
    def __init__(self, filters, use_bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=use_bias)
        self.b1 = InstanceNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=use_bias)
        self.b2 = InstanceNormalization()
        self.relu2 = ReLU()

    def call(self, x):
        y = self.conv1(x)
        y = self.b1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.b2(y)
        y = tf.keras.layers.Add()([x, y])
        y = self.relu2(y)
        return y

def reflection_pad_2d(x, padding_size):
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='REFLECT')

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(channels // reduction_ratio, activation='relu', use_bias=False)
        self.fc2 = Dense(channels, activation='sigmoid', use_bias=False)
        self.reshape = Reshape((1, 1, channels))

    def call(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.reshape(y)
        return Multiply()([x, y])

class Generator(tf.keras.Model):
    def __init__(self, ngf, numresblock):
        super(Generator, self).__init__()
        self.pad = tf.keras.layers.Lambda(lambda x: reflection_pad_2d(x, 3))
        self.conv1 = Conv2D(filters=ngf, kernel_size=7, strides=1, padding='valid', use_bias=True)
        self.b1 = InstanceNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv2D(filters=ngf*2, kernel_size=3, strides=2, padding='same', use_bias=True)
        self.b2 = InstanceNormalization()
        self.relu2 = ReLU()

        self.conv3 = Conv2D(filters=ngf*4, kernel_size=3, strides=2, padding='same', use_bias=True)
        self.b3 = InstanceNormalization()
        self.relu3 = ReLU()

        self.numresblock = numresblock
        for i in range(numresblock):
            setattr(self, 'res'+str(i+1), ResBlock(ngf*4, use_bias=True))


        self.ca1 = ChannelAttention(ngf*4)
        self.ca2 = ChannelAttention(ngf*4)

        self.deconv3 = Conv2DTranspose(filters=ngf*2, kernel_size=3, strides=2, padding='same', use_bias=True)
        self.b4 = InstanceNormalization()
        self.relu4 = ReLU()

        self.deconv2 = Conv2DTranspose(filters=ngf, kernel_size=3, strides=2, padding='same', use_bias=True)
        self.b5 = InstanceNormalization()
        self.relu5 = ReLU()

        self.pad2 = tf.keras.layers.Lambda(lambda x: reflection_pad_2d(x, 3))
        self.deconv1 = Conv2D(filters=3, kernel_size=7, strides=1, padding='valid', use_bias=False)
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, img, pose):
        x = tf.concat([img, pose], axis=-1)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.b1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        x = self.b2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.b3(x)
        x = self.relu3(x)

        for i in range(self.numresblock):
            res = getattr(self, 'res'+str(i+1))
            x = res(x)


        x = self.ca1(x)
        x = self.ca2(x)

        x = self.deconv3(x)
        x = self.b4(x)
        x = self.relu4(x)

        x = self.deconv2(x)
        x = self.b5(x)
        x = self.relu5(x)

        x = self.pad2(x)
        x = self.deconv1(x)
        x = self.tanh(x)

        return x


gen = Generator(ngf=64, numresblock=9)
output = gen(img=tf.ones((1, 256, 128, 3)), pose=tf.ones((1, 256, 128, 3)))
gen.summary()


# Load the model weights
gen.load_weights('G_9.weights.h5')

# Generate the image
genimage = gen(img, pose)

# Squeeze the tensor to remove the batch dimension and convert it to NumPy
genimage_tensor = tf.squeeze(genimage, axis=0)
genimage_np = genimage_tensor.numpy() * 255
genimage_np = genimage_np.astype('uint8')

# Save the generated image using OpenCV
output_image_path = 'generated_image.png'  # Specify the file name and path
cv2.imwrite(output_image_path, cv2.cvtColor(genimage_np, cv2.COLOR_RGB2BGR))

# Optionally, display the image (you can remove this if not needed)
plt.imshow(genimage_np)
plt.axis('off')
plt.show()

print(f"Generated image saved at {output_image_path}")


import numpy as np
import tensorflow as tf

def extract_features(models, images):
    features = []
    if isinstance(images, str):
        img = tf.keras.preprocessing.image.load_img(images, target_size=(256, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        images = img_array
    for model in models:
        feature = model.predict(images)
        features.append(feature)
    return np.array(features)

def fuse_features(query_features, synthesized_features, query_weight=0.8, synthesized_weight=0.2):
    assert query_weight + synthesized_weight == 1.0, "The weights must sum to 1.0"
    fused_features = query_weight * query_features + synthesized_weight * synthesized_features
    return fused_features

def compute_euclidean_distance(query_features, gallery_features):
    distances = np.linalg.norm(gallery_features - query_features, axis=1)
    return distances

def rank_gallery(distances):
    return np.argsort(distances)

def compare_with_gallery(query_features, gallery_features, synthesized_features):
    combined_features = np.concatenate([gallery_features, synthesized_features], axis=0)

    distances = compute_euclidean_distance(query_features, combined_features)

    ranking = rank_gallery(distances)

    match_index = np.argmin(distances)

    return ranking, match_index


model_a = tf.keras.models.load_model('reid3modelcuhk_a.h5')
model_b = tf.keras.models.load_model('reid3modelcuhk_b.h5')
model_c = tf.keras.models.load_model('reid2modelcuhk_c.h5')

# Define your local image paths (update these to your local paths)
gallery_image_path = 'testdir copy 2/label1/cam2/0014_c2_132.png.png'  # Replace with local path
synthesized_image_path = 'generated_image.png'  # Replace with local path
query_image_path = 'testdir copy 2/label1/cam1/0014_c1_127.png.png' # Replace with local path

query_features = extract_features([model_a], query_image_path)
synthesized_features = extract_features([model_b], synthesized_image_path)
gallery_features = extract_features([model_c], gallery_image_path)

fused_query_features = fuse_features(query_features, synthesized_features, query_weight=0.7, synthesized_weight=0.3)

ranking, match_index = compare_with_gallery(fused_query_features, gallery_features, synthesized_features)


print("Ranking of the combined features with respect to the query image:")
print(ranking)
print(f"The closest match index is: {match_index}")


def is_same_person(query_features, gallery_features, threshold=0.5):
    """
    Compare the query image features with gallery features and decide if they represent the same person
    based on a distance threshold.
    """

    distance = compute_euclidean_distance(query_features, gallery_features)


    is_same = distance < threshold
    return is_same, distance


distance_threshold = 0.39


is_same, distance = is_same_person(fused_query_features, gallery_features, threshold=distance_threshold)


if is_same:
    print(f"The query and (gallery,gen) images are of the same person. Distance: {distance}")
else:
    print(f"The query and (gallery,gen) images are of different persons. Distance: {distance}")






