import cv2
import matplotlib.pyplot as plt
import os
 
#이미지 디렉토리까지가서 읽기.
path = "/home/geonwoo/Documents/Rist/data/Part2"
os.chdir(path)
files = os.listdir(path)


#값 입력받기
q_x , q_y = input("쿼리의 위치를 알려주세요 : ").split()
query_cor = []
query_cor.append(q_x)
query_cor.append(q_y)

#DB좌표 만들기.
cor_list = []
query_color = []
i = 0
for data in files :
    #.jpg없에기
    data = data.replace('.JPG','')
    tmp = data.split('_')
    if tmp == query_cor :
        query_index = i
        query_color.append(30)
    else :
        query_color.append(0)
    cor_list.append([])
    for t in tmp :
        t = int(t)
        cor_list[i].append(t)
    i = i + 1

#query 좌표와 함께 plot그리기
x, y = zip(*cor_list)
plt.scatter(x, y,30,query_color)
plt.show()

