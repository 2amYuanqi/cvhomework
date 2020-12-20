import cv2 as cv
import numpy as np
from matplotlib.pylab import *

def cornerHarris(img, blocksize=2, ksize=3, k=0.04): # ksize是sobel的大小，blocksize是滑动窗口的大小
    def _clacHarris(cov, k):
        result = np.zeros([cov.shape[0],cov.shape[1]], dtype=np.float32)
        for i in range(cov.shape[0]):     #求出Harris矩阵
            for j in range(cov.shape[1]):
                a = cov[i, j, 0]
                b = cov[i, j, 1]
                c = cov[i, j, 2]
                result[i,j] = a * c - b* b - k * (a+c) *(a+c)  #计算每个像素的harris角点响应值
        return result

    #用sobel求x、y方向上的梯度
    Dx = cv.Sobel(img, cv.CV_32F,1,0,ksize = ksize) #对x求梯度
    Dy = cv.Sobel(img, cv.CV_32F,0,1,ksize = ksize) #对y求梯度

    #求harris矩阵
    cov = np.zeros([img.shape[0],img.shape[1],3], dtype = np.float32) 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cov[i,j,0] = Dx[i,j] * Dx[i,j] 
            cov[i,j,1] = Dx[i,j] * Dy[i,j]
            cov[i,j,2] = Dy[i,j] * Dy[i,j]

    #cov = cv.GaussianBlur(cov,(blocksize,blocksize),1)  #窗口用高斯函数
    cov = cv.boxFilter(cov, -1, (blocksize, blocksize), normalize = False)

    return _clacHarris(cov,k)

def corner_detect(img, min=20, threshold=0.04):
    # 首先对图像进行阈值处理
    _threshold = img.max() * threshold
    threshold_img = (img > _threshold) * 1
    coords = np.array(threshold_img.nonzero()).T
    candidate_values = [img[c[0], c[1]] for c in coords]
    index = np.argsort(candidate_values)

    neighbor = np.zeros(img.shape)
    neighbor[min:-min, min:-min] = 1
    filtered_coords = []
    for i in index:
        if neighbor[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            neighbor[(coords[i, 0] - min):(coords[i, 0] + min),
            (coords[i, 1] - min):(coords[i, 1] + min)] = 0
    return filtered_coords

def corner_plot(image, filtered_coords):
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'ro')
    axis('off')
    show()

if __name__ == "__main__":
    img = cv.imread("test.jpg")
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    result = cornerHarris(gray_img,2,3,0.04)
    corner_img = corner_detect(result, min=15, threshold=0.04)
    corner_plot(img, corner_img)
    #cv.waitKey(0)

