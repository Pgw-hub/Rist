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

#input answer_query
answer = int(input("디비와 매칭시킬 퀴리이미지의 인덱스를 입력하시오 "))

#trakcing
query_cor_list = []
query_color = []
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
    if curr_rot_flag == True and flags == 0 : rot_index = i
  
    translation_xy = track.get_translation(t, translation_xy, flags)
    
    int_traslation_xy = [int(x) for x in translation_xy]
    # print("real query_coordinate", int_traslation_xy)
    query_cor_list.append(int_traslation_xy)
    
    prev_rot_flag = curr_rot_flag
    if i == answer : 
        query_color.append(251)
    else : 
        query_color.append(0)

    
    # plt.scatter(translation_xy[0], translation_xy[1])
    # plt.annotate(str(i),xy=(translation_xy[0],translation_xy[1]), xytext=(translation_xy[0]+0.05,translation_xy[1]+0.05))


#query_normalize
x, y = zip(*query_cor_list)
query_norm_x, query_norm_y = track.normalize(x, y)
#쿼리 좌표 처리.
for i in range(0,len(query_norm_x)) :
    if i >= rot_index : #회전 이후
        query_norm_x[i] = query_norm_x[i] * 0.95
        query_norm_y[i] = 1.0
    else : #회전 이전
        query_norm_x[i] = 0.0
        query_norm_y[i] = query_norm_y[i] * 1.6

#norm_query의 answer좌표 받아와서 dbmap에 넘기기.
answer_cor = []
for i in range(0,len(query_norm_x)) :
    if i == answer - 1 :
        answer_cor.append(query_norm_x[i])
        answer_cor.append(query_norm_y[i])
print("answer_cor = ",answer_cor)

#db_normalize and plot
track.db_map(answer_cor)

# print("Query norm cor_x = ", query_norm_x)
# print()
# print("Query norm cor_y = ", query_norm_y)
# print()
# print("Number of Query",len(query_norm_x))
plt.scatter(query_norm_x,query_norm_y,10,query_color)

plt.show()