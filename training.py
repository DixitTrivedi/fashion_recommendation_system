import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
import pickle
import os
from tqdm import tqdm


def load_resnet50_model():
    """Loads the pre-trained ResNet50 model with frozen weights and global max pooling."""
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPool2D()])
    return model


def preprocess_image(img_path, model):
    """Loads, resizes, normalizes, and preprocesses an image for feature extraction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    ex_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(ex_img_arr)
    return preprocessed_img


def extract_features(image_paths, model):
    """Extracts features from a list of image paths using the preprocessed model."""
    feature_list = []
    for img_path in tqdm(image_paths):
        preprocessed_img = preprocess_image(img_path, model)
        feature = model.predict(preprocessed_img).flatten()
        normalized_feature = feature / norm(feature)
        feature_list.append(normalized_feature)
    return feature_list


def save_features(features, filepath):
    """Saves extracted features to a pickle file."""
    pickle.dump(features, open(filepath, 'wb'))


def main():
    """Loads the model, processes images, extracts features, and saves them."""

    # Load pre-trained ResNet50 model
    model = load_resnet50_model()

    # Get working directory
    working_dir = os.getcwd()

    # Image path
    image_dir = os.path.join(working_dir, 'archive/images')

    # List image paths
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]

    # Save image paths for future reference (optional)
    pickle.dump(image_paths, open(os.path.join(working_dir, 'artifacts/images.pkl'), 'wb'))

    # Extract features
    features = extract_features(image_paths, model)

    # Save features
    save_features(features, 'features.pkl')

    print("Feature extraction complete!")


if __name__ == '__main__':
    main()
