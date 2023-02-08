# import cv2
# import matplotlib.pyplot as plt
# import os
 
# #이미지 디렉토리까지가서 읽기.
# path = "/home/sungmin/바탕화면/CGV/datata/"
# os.chdir(path)
# files = os.listdir(path)


# #값 입력받기
# q_x , q_y = input("쿼리의 위치를 알려주세요 : ").split()
# query_cor = []
# query_cor.append(q_x)
# query_cor.append(q_y)

# #DB좌표 만들기.
# cor_list = []
# query_color = []
# i = 0
# for data in files :
#     #.jpg없에기
#     data = data.replace('.JPG','')
#     tmp = data.split('_')
#     if tmp == query_cor :
#         query_index = i
#         query_color.append(30)
#     else :
#         query_color.append(0)
#     cor_list.append([])
#     for t in tmp :
#         t = int(t)
#         cor_list[i].append(t)
#     i = i + 1

# #query 좌표와 함께 plot그리기
# x, y = zip(*cor_list)
# plt.scatter(x, y,30,query_color)
# plt.show()

#############
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

#이미지 디렉토리까지가서 읽기.
path = "/home/sungmin/바탕화면/CGV/datata/"
os.chdir(path)
files = os.listdir(path)


#값 입력받기 10->11
#q_x , q_y = input("쿼리의 위치를 알려주세요 : ").split()
q_x = 73.08 
q_y = 11.93

q_x = round(q_x/13, 1)
q_y = round(q_y/25)


query_cor = []
query_cor.append(q_x)
query_cor.append(q_y)

#값 입력받기2 11->12
#q_x1 , q_y1 = input("쿼리의 위치를 알려주세요 : ").split()
q_x1 = 87.69
q_y1 = 14.32

q_x1 = round(q_x1/13, 1)
q_y1 = round(q_y1/250)


query_cor1 = []
query_cor1.append(q_x1)
query_cor1.append(q_y1)


#DB좌표 만들기.
cor_list = []
x_list = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
y_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12]
# query_color = []
i = 0
for data in files :
    #.jpg없에기
    data = data.replace('.JPG','')
    tmp = data.split('_')
    # if tmp == query_cor :
    #     query_index = i
    #     query_color.append(30)
    # else :
    #     query_color.append(0)
    cor_list.append([])
    for t in tmp :
        t = int(t)
        cor_list[i].append(t)
    i = i + 1

#query 좌표와 함께 plot그리기
x, y = zip(*cor_list)

plt.figure(figsize=(15,6))
plt.scatter(x, y,30,c='blue')
plt.scatter(q_y, q_x, 40, c='green')
print(q_y, '/' , q_x)
plt.scatter(q_y1, q_x1, 40, c='yellow')
print(q_y1, '/', q_x1)
plt.xticks(x_list, rotation = 45)
plt.yticks(y_list)
plt.show()
#65.7722760731855 ,  10.740554344020918
#73.08030674798876 ,  11.933949271136271