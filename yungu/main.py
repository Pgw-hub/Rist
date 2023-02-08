import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import tracking as track

num_images = 55
images = []

for i in range(0, num_images):
    img = cv2.imread("/home/geonwoo/Documents/Rist/data/query/Part2_undistort/undistort_" + str(i) + ".jpg")
    images.append(img)

# undistortion images camera intrinsic parameter
K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])

E_update = np.eye(3)
translation_xy = [0.0, 0.0]

nfeatures = 1000

prev_rot_flag = False
flags = 1

track.db_map()
query_cor_list = []
for i in range(1, num_images):
    
    #################### Method 1 #########################
    src_pts, dst_pts = track.calc_optical_flow(images[i-1], images[i])
    
    # #################### Method 2 #########################
    # kp1, des1 = track.feature_detection(images[i-1], nfeatures)
    # kp2, des2 = track.feature_detection(images[i], nfeatures)
    
    # src_pts, dst_pts, good_matches = track.feature_matching(kp1, kp2, des1, des2, "BF")
    
    
    mean_direction_vector = track.get_mean_direction_vector(src_pts, dst_pts)
    
    angle = track.get_angle(mean_direction_vector)
    
    R, t, E_update = track.get_Rt(src_pts, dst_pts, 301, (316., 251.), E_update)
   
    curr_rot_flag, flags = track.detect_rotation(prev_rot_flag, flags, angle, 80)
  
    translation_xy = track.get_translation(t, translation_xy, flags)
    int_traslation_xy = [int(x) for x in translation_xy]
    query_cor_list.append(int_traslation_xy)
    
    prev_rot_flag = curr_rot_flag
    
    plt.scatter(translation_xy[0], translation_xy[1])
    plt.annotate(str(i),xy=(translation_xy[0],translation_xy[1]), xytext=(translation_xy[0]+0.05,translation_xy[1]+0.05))
print(query_cor_list)
x, y = zip(*query_cor_list)
print("query cor", x)
query_norm_x, query_norm_y = track.normalize(x, y)
plt.scatter(query_norm_x,query_norm_y,10,c = 'green')
plt.show()