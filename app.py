import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pickle
import os
from PIL import Image


st.set_page_config(layout="wide", page_icon="image", page_title="Fashion Recommendation System")
st.title('Fashion Recommendation System')


def load_model():
    """Loads the pre-trained ResNet50 model with frozen weights and global max pooling."""
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPool2D()])
    return model


def load_features_and_images():
    """Loads stored image paths and features from pickle files."""
    with open(os.path.join(os.getcwd(), 'artifacts','images.pkl'), 'rb') as f:
        file_img = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'artifacts','features.pkl'), 'rb') as f:
        feature_list = pickle.load(f)
    return file_img, feature_list


def save_uploaded_image(upload_img):
    """Saves an uploaded image to the 'uploads' directory."""
    try:
        with open(os.path.join(os.getcwd(), 'artifacts/uploads', upload_img.name), 'wb') as f:
            f.write(upload_img.getbuffer())
        return True
    except Exception as e:
        print(f'Error saving uploaded image {str(e)}')
        return False


def extract_features(image_path, model):
    """Extracts features from an image using the pre-trained model."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    ex_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(ex_img_arr)
    feature = model.predict(preprocessed_img).flatten()
    normalized_feature = feature / norm(feature)
    return normalized_feature


def recommend_products(features, feature_list):
    """Recommends similar products using nearest neighbors with Euclidean distance."""
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    knn.fit(feature_list)
    distances, indices = knn.kneighbors([features])
    return indices.flatten()


def display_uploaded_images(upload_images):
    """Displays uploaded images in a grid layout."""
    if upload_images:
        cols = st.columns(10)
        for i, col in enumerate(cols):
            try:
                col.image(Image.open(upload_images[i]), use_column_width="always")
            except:
                pass
        st.markdown("---")


def display_recommendations(recommend_indices, file_images):
    """Displays recommended product images in a grid layout."""
    if recommend_indices:
        st.header("Similar Products")
        a, b = 0, 5
        for _ in range(int(len(recommend_indices) / 5)):
            im_cols = st.columns(b - a)
            for i, col in enumerate(im_cols, start=a):
                col.image(Image.open(file_images[recommend_indices[i]]))
            a, b = b, b + 5


def main():
    """Loads the model, features, images, handles uploads, and displays recommendations."""
    model = load_model()
    file_images, feature_list = load_features_and_images()

    uploaded_images = st.file_uploader("Choose an image", accept_multiple_files=True)

    display_uploaded_images(uploaded_images)

    if uploaded_images:
        recommendations = []
        for image in uploaded_images:
            if save_uploaded_image(image):
                features = extract_features(os.path.join(os.getcwd(), 'artifacts/uploads', image.name), model)
                indices = recommend_products(features, feature_list)
                recommendations.append(indices)

        recommend_indices = [item for sublist in recommendations for item in sublist.flatten()]
        display_recommendations(recommend_indices, file_images)


if __name__ == '__main__':
    main()
