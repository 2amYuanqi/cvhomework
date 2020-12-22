import pandas as pd
import numpy as np
from cv2 import cv2


img = cv2.imread("1.jpg")
print(([img,img,img],[img,img,img])[0][0].shape)
