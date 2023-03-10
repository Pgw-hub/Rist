import cv2
import numpy as np
import matplotlib.pyplot as plt
import math




    
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
    
    if flags == 0:
        translation_xy[0] += x_translation
        translation_xy[1] += y_translation
    else:
        translation_xy[0] += y_translation
        translation_xy[1] += x_translation
        
    return translation_xy

