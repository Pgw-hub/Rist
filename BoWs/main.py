import cv2
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import argparse
import backofwords as bow

#인자 파싱
parser = argparse.ArgumentParser(description='Generate or load vocabulary.') 
parser.add_argument('--mode', type = str, default='generate', help = 'generate or load')
parser.add_argument('--vocab_path', type = str, default = 'dictionary/vocabulary.npy', help = 'path to vocabulary file')

args = parser.parse_args()
mode = args.mode
vocab_path = args.vocab_path


start_time = time.time()
#이미지 읽어오기
image_files = [filename for filename in os.listdir('test') if filename.endswith('.JPG')]
images = [cv2.imread(os.path.join('test', filename)) for filename in image_files]

k = 50
#vocabulary 설정
if mode == 'generate' :
    vocabulary = bow.build_vocabulary(images, k)
    np.save('dictionary/vocabulary.npy', vocabulary)
elif mode == 'load' :  
    vocabulary = np.load('dictionary/vocabulary.npy')

#쿼리 읽어와서 가장 비슷한 거 비교하기.
query_image = cv2.imread("query/undistorted_mid.jpg")
similar_image = bow.find_similar_image(query_image, images, vocabulary)
end_time = time.time()
elapsed_time = end_time - start_time

#결과 표시
print("elapsed_time : ",elapsed_time)
similar_image = cv2.resize(similar_image,(640,480))
query_image = cv2.resize(query_image,(640,480))
cv2.imshow("similar_img in DB", similar_image)
cv2.imshow("query_img",query_image)
cv2.waitKey(0)
