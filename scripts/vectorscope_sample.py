import os
import sys
import cv2

sys.path.append(os.getcwd())
import vectorscope  # noqa E402


if __name__ == '__main__':
    if sys.platform == "win32":
        img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    elif sys.platform == "linux":
        img = cv2.imread("/home/jjj/Pictures/MIN-2K4-KNA_A_FBL.jpg")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cb, cr, colors = vectorscope.extractCbCrData(rgbImg)
    vectorscopeImg = vectorscope.createVectorscopeImg(cb, cr, colors)
    cv2.imshow('img', vectorscopeImg)
    cv2.waitKey(0)
