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
def is_same_person(query_features, gallery_features, threshold=0.5):
    """
    Compare the query image features with gallery features and decide if they represent the same person
    based on a distance threshold.
    """

    distance = compute_euclidean_distance(query_features, gallery_features)


    is_same = distance < threshold
    return is_same, distance


