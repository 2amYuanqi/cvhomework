from cv2 import cv2
import numpy as np
import pandas as  pd

width = 1280
height = 720
cap = cv2.VideoCapture(0) #打开摄像头
#设置宽和高
cap.set(3,width)
cap.set(4,height)
#cap.set(10,130)

##标出图像轮廓
def preProcessing(img):
    '''
    功能：标出图像轮廓
    返回：标出轮廓的图像
    '''
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    #对边缘进行膨胀
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres
    #return imgCanny

##找出最大轮廓和最大轮廓的折点
def getContours(img):
    '''
    功能：找出最大轮廓和最大轮廓的折点,并在图像上画出
    返回：最大轮廓的折点
    '''
    biggestCorner = np.array([]) #用于存放最大轮廓的角点
    maxArea = 0
    #找出所有的轮廓
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours: #遍历轮廓
        area = cv2.contourArea(cnt)  #计算面积
        if area >5000:
            #在imgcontour上画出cnt这个轮廓
            #cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            #计算cnt的周长
            peri = cv2.arcLength(cnt,True)
            #对轮廓进行多边拟合，得到拟合的折点
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)#估计点的数量有周长*0.02个
            if area > maxArea and len(approx) == 4: #要检测的为四边形，所以轮廓的折点要为4
                biggestCorner = approx #将最大轮廓的折点赋给biggestContour
                maxArea = area
    #绘出四个点
    #cv2.drawContours(imgContour, biggestCorner,-1,(255,0,0),20)
    return biggestCorner

##对轮廓的四个角点进行排序
def reorder(myPoints):
    myPoints = myPoints.reshape(4,2)#四行：4个点；两列：x，y
    myPointsNew = np.zeros((4,1,2),np.int32)#变成三维，4个点；每个里面是1个点，包含x、y信息
    add = myPoints.sum(1)#axis = 1,列相加，把每个点的x、y加起来
    myPointsNew[0] = myPoints[np.argmin(add)]#xy之和最小的为左上角的点
    myPointsNew[3] = myPoints[np.argmax(add)]#xy之和最大的为右下角的点
    diff = np.diff(myPoints,axis = 1) #y-x 的绝对值
    myPointsNew[1] = myPoints[np.argmin(diff)]#y-x的绝对值最小的点为右上角
    myPointsNew[2] = myPoints[np.argmin(diff)]#y-x的绝对值最大的点为左下角
    return myPointsNew#顺序：左上、右上、左下、右下

##透视变换
def getWrap(img,biggestCorner):
    '''
    功能：对输入图像进行透视变换
    返回：变换后的图像
    '''
    
    biggestCorner = reorder(biggestCorner)
    pts1 = np.float32(biggestCorner)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutPut = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    imgCropped = imgOutPut[20:imgOutPut.shape[0]-20,20:imgOutPut.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))
    return imgCropped

while True:
    widthImg,heightImg = 1280,720
    success,img = cap.read()

    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()
    imgthred = preProcessing(img)
    getContours(imgthred)
    biggestCorner = getContours(imgthred)
    

    if biggestCorner.size!= 0:
        imgWrapped = getWrap(img,biggestCorner)
        cv2.imshow("1",imgWrapped)
    else:
        cv2.imshow("1",imgContour)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
