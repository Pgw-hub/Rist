import cv2
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

def sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def build_vocabulary(images, k):
    descriptors = []
    for image in images :
        keypoints, descriptor = sift_features(image)
        descriptors.append(descriptor)
    descriptors = np.vstack(descriptors)
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(descriptors)
    vocabulary = kmeans.cluster_centers_
    return vocabulary

def find_similar_image(query_image, images, vocabulary): 
    similarities = []
    for i, image in enumerate(images) :
        similaritiy = compute_image_similarity(query_image, image, vocabulary)
        similarities.append((i,similaritiy))

    similarities = sorted(similarities, key = lambda x : x[1])
    return images[similarities[0][0]]

def get_bow_histogram(descriptors, vocabulary) : 
    k = len(vocabulary)
    bow_histogram = np.zeros(k)
    for descriptor in descriptors :
        distances = np.linalg.norm(vocabulary - descriptor, axis = 1)
        closest_cluster_index = np.argmin(distances)
        bow_histogram[closest_cluster_index] += 1
    
    return bow_histogram

def normalize_histogram(histogram):
    norm = np.linalg.norm(histogram)
    if norm == 0:
        return histogram
    return histogram / norm

def compute_image_similarity(image1, image2, vocabulary) :
    k = len(vocabulary)
    image1_keypoints, image1_descriptors = sift_features(image1)
    image2_keypoints, image2_descriptors = sift_features(image2)
    image1_histogram = get_bow_histogram(image1_descriptors, vocabulary) 
    image2_histogram = get_bow_histogram(image2_descriptors, vocabulary)  
    image1_histogram = normalize_histogram(image1_histogram)
    image2_histogram = normalize_histogram(image2_histogram)
    distances = np.linalg.norm(image1_histogram - image2_histogram)
    return distances