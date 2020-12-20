import numpy as np
import cv2 as cv
import random as rd

cap = cv.VideoCapture(0)#通过本地摄像头捕获视频
fourcc = cv.VideoWriter_fourcc(*'DIVX')#指定fourcc编码
out = cv.VideoWriter('snow.mp4',fourcc,20.0,(480,480))
'''

这里，out是一个VideoWriter实例化对象，第一个参数是要制作的视频的文件名，fourcc后面会详解，20是fps，
接下来是视频的长宽，如果要保存只有两维的灰度图，则最后还要加个False或者0，不添加默认是彩色。
'''


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)     # 高斯模糊
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edge_output = cv.Canny(gray, 50, 100)
    return edge_output

while cap.isOpened():
    ret, frame = cap.read()
    '''
    cap.read()是按帧读取，返回两个值：ret,frame
    ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False；
    后面的frame该帧图像的三维矩阵BGR形式。
    '''
    if not ret:
        print('Can’t recive frame(stream end?). Exiting ...')
        break

    frame = cv.resize(frame,(480,480)) 

    edge = edge_demo(frame)
    
    ac_list = []  
    for i in range(200):
        ac_list.append(rd.randint(1,2))

    x = []
    for i in range (200):
        x.append([rd.randint(0,479),rd.randint(0,479),4])
    #print(x)
    num = 0
    for i in x:
        a = i[0]
        b = i[1]
        if edge[a][b] == 255:
            cv.circle(frame,(i[0],i[1]),i[2],[255,255,255],thickness=cv.FILLED) #边缘

        else:
            i[0] = rd.randint(i[0]-ac_list[num],i[0]+ac_list[num]) #x
            i[1] = i[1] + ac_list[num]  #y
            num = num +1
            cv.circle(frame,(i[0],i[1]),i[2],[255,255,255],thickness=cv.FILLED)

        cv.imshow('snow', frame)#显示图像帧（将每一帧图像连续显示便是一段视频）
        

        #保存
        out.write(frame)
    if cv.waitKey(1) == ord('q'):#等待键盘响应，按下‘q’保存并退出
        break

#工作完成后，释放所有内容
cap.release()
out.release()
cv.destroyAllWindows()
#cv.waitKey(0)