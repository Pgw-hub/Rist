import cv2
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors
import math
import tracking as track


num_images = 55
images = []
# #undistort
# for i in range(0, num_images):
#     img = cv2.imread("/home/geonwoo/Documents/Rist/data/query/Part2_undistort/undistort_" + str(i) + ".jpg")
#     images.append(img)
# # undistortion images camera intrinsic parameter
# K = np.array([[301.39596558, 0.0, 316.70672662],
#                          [0.0, 300.95941162, 251.54445701],
#                          [0.0, 0.0, 1.0]])
#distort
for i in range(0, num_images):
    img = cv2.imread("/home/geonwoo/Documents/Rist/data/query/Part2_distort/" + str(i) + ".jpg")
    images.append(img)

# undistortion images camera intrinsic parameter
K = np.array([[301.867624408757, 0.0, 317.20235900477695],
                         [0.0, 301.58768437338944, 252.0695806789168],
                         [0.0, 0.0, 1.0]])
                         


E_update = np.eye(3)
translation_xy = [0.0, 0.0]
nfeatures = 1000
prev_rot_flag = False
flags = 1

#distort
x_max = 333
x_min = 1
y_max = 166
y_min = 7


# #undistort
# x_max = 321
# x_min = 1
# y_max = 137
# y_min = 7


# #db_normalize and plot
db_traslation_x, db_traslation_y = track.db_map()


#input answer_query, answer_range = 0
answer = int(input("디비와 매칭시킬 퀴리이미지의 인덱스를 입력하시오 "))
answer_range = float(input("Range : "))

#trakcing
rot_flag = False
answer_xy = []
for i in range(1, answer + 1):
    #################### Method 1 #########################
    src_pts, dst_pts = track.calc_optical_flow(images[i-1], images[i])
    
    mean_direction_vector = track.get_mean_direction_vector(src_pts, dst_pts)
    
    angle = track.get_angle(mean_direction_vector)
    
    R, t, E_update = track.get_Rt(src_pts, dst_pts, 301, (317., 252.), E_update)
   
    curr_rot_flag, flags = track.detect_rotation(prev_rot_flag, flags, angle, 80)
    if curr_rot_flag == True and flags == 0 : rot_flag = True
  
    translation_xy = track.get_translation(t, translation_xy, flags)
    
    int_traslation_xy = [int(x) for x in translation_xy]
    
    norm_int_translation_xy = track.normalize_query_point(int_traslation_xy, x_max, x_min, y_max, y_min)

    final_translation_xy = track.final_query_point(norm_int_translation_xy, rot_flag)
    
    prev_rot_flag = curr_rot_flag
    
    # if i == answer :
    plt.scatter(final_translation_xy[0],final_translation_xy[1], s = 60, c = 'red')

track.db_matching(db_traslation_x,db_traslation_y,final_translation_xy,answer_range)
plt.show()