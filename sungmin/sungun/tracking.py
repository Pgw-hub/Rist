import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import natsort

    
def feature_detection(img, nfeatures):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray1, None)
    return kp, des
    

def feature_matching(kp1, kp2, desc1, desc2, matcher):
    
    if matcher == "BF":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return src_pts, dst_pts, good_matches
        
        
    elif matcher == "FLANN":
        pass

def calc_optical_flow(img1, img2):
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    src_pts = prev_pts[status == 1]
    dst_pts = next_pts[status == 1]
    return src_pts, dst_pts
    

def get_mean_direction_vector(src_pts, dst_pts):
    motion_vectors = src_pts - dst_pts
    direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
    mean_direction_vector = np.mean(direction_vector, axis=0)
    mean_direction_vector = mean_direction_vector / np.linalg.norm(mean_direction_vector)
    return mean_direction_vector

def get_angle(direction):
    reference = (1, 0)
    dot_product = direction[0] * reference[0] + direction[1] * reference[1]
    angle = math.acos(dot_product)
    angle =  angle * 180 / math.pi
    return angle

def get_Rt(src_pts, dst_pts, focal_, pp_, E_update):
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=focal_, pp=pp_, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    E_update = np.dot(E_update,E)
    
    _, R, t, mask = cv2.recoverPose(E_update, src_pts, dst_pts, focal=focal_, pp=pp_)
    
    return R, t, E_update

def detect_rotation(prev_rot_flag, flags, angle, threshold=90):
    if angle > threshold:
        curr_rot_flag = True
    else:
        curr_rot_flag = False
    
    if prev_rot_flag == False and curr_rot_flag == True:
        if flags == 0: flags += 1
        else: flags -= 1
    
    return curr_rot_flag, flags
    
def get_translation(t, translation_xy, flags):
    x_translation = t[0][0] / t[2][0]
    y_translation = t[1][0] / t[2][0]
    
    if flags == 0: #회전 후
        translation_xy[0] += x_translation 
        translation_xy[1] += y_translation 
    else: #회전 전
        translation_xy[0] += y_translation
        translation_xy[1] += x_translation
        
    return translation_xy

def db_map():
    #이미지 디렉토리까지가서 읽기.
    path = "/home/geonwoo/Documents/Rist/data/Part2"
    os.chdir(path)
    files = os.listdir(path)
    files = natsort.natsorted(files)

    #DB좌표 만들기.
    cor_list = []
    i = 0
    for data in files :
        #.jpg없에기
        data = data.replace('.JPG','')
        tmp = data.split('_')
        cor_list.append([])
        for t in tmp :
            t = int(t)
            cor_list[i].append(t)
        i = i + 1

    x, y = zip(*cor_list)
    x ,y = normalize(x,y)

    x = [x[i] * 5 for i in range(len(x))]

    plt.figure(figsize=(18, 6))
    plt.scatter(x,y,100, c='blue')
    return x, y

def db_matching(db_traslation_x, db_traslation_y, answer_cor, answer_range) :
    x_range_left = answer_cor[0] - answer_range
    x_range_right = answer_cor[0] + answer_range
    y_range_bottom = answer_cor[1] - answer_range
    y_range_top =  answer_cor[1] + answer_range

    db_answer_index = []
    for i in range(0,len(db_traslation_x)) :
        if x_range_left < db_traslation_x[i] and db_traslation_x[i] < x_range_right and y_range_bottom < db_traslation_y[i] and db_traslation_y[i] < y_range_top :
            db_answer_index.append(i)
            
    print("matching DB index is",db_answer_index)
    for i in db_answer_index :
        print(i , " = ", db_traslation_x[i], " , ", db_traslation_y[i])

def normalize(cor_x,cor_y):
    x = list(cor_x)
    y = list(cor_y)
    tmp_max = max(x)
    tmp_min = min(x)
    for i in range(len(x))  :
        x[i] = (x[i] - tmp_min) / (tmp_max - tmp_min)
    
    tmp_max = max(y)
    tmp_min = min(y)
    for i in range(len(y))  :
        y[i] = (y[i] - tmp_min) / (tmp_max - tmp_min)

    return x,y

def normalize_query_point(int_translation_xy,x_max,x_min,y_max,y_min):
    int_translation_xy[0] = (int_translation_xy[0] - x_min) / (x_max - x_min)
    int_translation_xy[1] = (int_translation_xy[1] - y_min) / (y_max - y_min)
    return int_translation_xy

def final_query_point(norm_int_traslation_xy, rot_flag) :
    if rot_flag == True : #회전 이후
        norm_int_traslation_xy[0] = (norm_int_traslation_xy[0] - 0.085) * 5.45 # 0.085 -> x = 0에서 시작하게, 5.45는 search range를 위해 늘리기
        norm_int_traslation_xy[1] = 1.0
    else : #회전 이전
        norm_int_traslation_xy[0] = 0.0
        norm_int_traslation_xy[1] = norm_int_traslation_xy[1] * 1.9
    
    return norm_int_traslation_xy
        
def check_gap(forgap, rot_index = 1) : #checking x_cor gap
    prev = forgap[rot_index - 1]
    for i in range(1, len(forgap)) :
        if i >= rot_index :
            if i == rot_index : prev = forgap[i][0]
            else :
                curr = forgap[i][0]
                gap = prev - curr
                print("gap(", i - 1, " , ",i, ") = ", gap)
                prev = curr
