from cv2 import cv2
import numpy as np 

##
def stackImages(scale,imgArray):
    '''
    功能：拼凑图片
    返回：拼凑好的图片
    参数：
        scale：图片缩放的尺度
        imgArray：元组，要拼凑图片的矩阵
    '''
    rows = len(imgArray) #图片矩阵的行数
    cols = len(imgArray[0])#图片矩阵的列数
    rowsAvailable = isinstance(imgArray[0],list)#判断是否为list，是为true
    #第一张图片的宽度和高度
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    #依次对每张图片进行缩放,
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:  #该张图片与第一张图片的大小一样，直接将该图缩放scale
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                else:   #该张图片与第一张图片的大小不一样，先将该图转换为第一张图片的大小，再将该图缩放scale
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)
                if len(imgArray[x][y].shape) == 2:#如果是灰度图,转换为bgr图
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height,width,3),np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0,rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].sahpe[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)
            if len(imgArray[x].shape) == 2 :
                imgArray[x] = cv2.cvtClolor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread("1.jpg")
#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgStack = stackImages(0.5,([img,img,img],[img,img,img]))
cv2.imshow(".",imgStack)
cv2.waitkey(0)

            

