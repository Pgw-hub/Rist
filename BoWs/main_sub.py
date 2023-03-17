import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def sift_features(image):
    """이미지에서 SIFT 특징점 및 기술자 추출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def build_vocabulary(descriptors, k):
    """K-Means 클러스터링을 통한 사전 구성"""
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(descriptors)
    vocabulary = kmeans.cluster_centers_
    return vocabulary


def get_bow_histogram(descriptors, vocabulary):
    """Bag-of-Words 히스토그램 생성"""
    k = len(vocabulary)
    bow_histogram = np.zeros(k)
    for descriptor in descriptors:
        distances = np.linalg.norm(vocabulary - descriptor, axis=1)
        closest_cluster_index = np.argmin(distances)
        bow_histogram[closest_cluster_index] += 1
    return bow_histogram


def normalize_histogram(histogram):
    """L2 노름으로 정규화된 히스토그램 반환"""
    norm = np.linalg.norm(histogram)
    if norm == 0:
        return histogram
    return histogram / norm


def compute_image_similarity(image1, image2, vocabulary):
    """두 이미지 간의 유사도 계산"""
    k = len(vocabulary)
    image1_keypoints, image1_descriptors = sift_features(image1)
    image2_keypoints, image2_descriptors = sift_features(image2)
    image1_histogram = get_bow_histogram(image1_descriptors, vocabulary)
    image2_histogram = get_bow_histogram(image2_descriptors, vocabulary)
    image1_histogram = normalize_histogram(image1_histogram)
    image2_histogram = normalize_histogram(image2_histogram)
    distances = np.linalg.norm(image1_histogram - image2_histogram)
    return distances


# 이미지 불러오기
image1 = cv2.imread("data/image1.jpg")
image2 = cv2.imread("data/image2.jpg")

# 이미지에서 SIFT 특징점 및 기술자 추출
image1_keypoints, image1_descriptors = sift_features(image1)
image2_keypoints, image2_descriptors = sift_features(image2)

# K-Means 클러스터링을 통한 사전 구성
k = 10
vocabulary = build_vocabulary(np.vstack((image1_descriptors, image2_descriptors)), k)

# 이미지들의 Bag-of-Words 히스토그램 생성
image1_histogram = get_bow_histogram(image1_descriptors, vocabulary)
image2_histogram = get_bow_histogram(image2_descriptors, vocabulary)

# L2 노름으로 정규화된 히스토그램 반환
image1_histogram = normalize_histogram(image1_histogram)
image2_histogram = normalize_histogram(image2_histogram)

# 이미지 간 유사도 계산
similarity = compute_image_similarity(image1, image2, vocabulary)
print("Image similarity:", similarity)
