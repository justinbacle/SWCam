import cv2
import numpy as np
import numba


def extractCbCrData(rgbImg: np.ndarray) -> np.ndarray:

    if rgbImg.dtype == np.uint8:
        rgbImg = rgbImg.astype(np.float32) / 255
    elif rgbImg.dtype == np.uint16:
        rgbImg = rgbImg.astype(np.float32) / np.max(rgbImg)

    yCrCbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2YCrCb)
    crData = yCrCbImg[:, :, 1].flatten()
    cbData = yCrCbImg[:, :, 2].flatten()

    yNull = np.ones_like(yCrCbImg[:, :, 0]) / 2

    fakeYCrCbImg = np.zeros_like(rgbImg)
    fakeYCrCbImg[:, :, 0] = yNull
    fakeYCrCbImg[:, :, 1] = yCrCbImg[:, :, 1]
    fakeYCrCbImg[:, :, 2] = yCrCbImg[:, :, 2]
    fakeRgbImg = cv2.cvtColor(fakeYCrCbImg, cv2.COLOR_YCrCb2RGB)

    fakeRData = fakeRgbImg[:, :, 0].flatten()
    fakeGData = fakeRgbImg[:, :, 1].flatten()
    fakeBData = fakeRgbImg[:, :, 2].flatten()

    colors = []
    for i in range(len(crData)):
        colors.append([fakeRData[i], fakeGData[i], fakeBData[i]])
    colors = np.clip(colors, 0, 1)

    return (cbData, crData, colors)


@numba.njit()
def createVectorscopeImg(cb, cr, colors, width=360, height=360):
    colors = colors*256
    vectorscopeImg = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(len(cb)):
        vectorscopeImg[int(cb[i]*width), int(cr[i]*height), :] += colors[i]/10
    vectorscopeImg = np.log(vectorscopeImg)
    vectorscopeImg = vectorscopeImg/np.max(vectorscopeImg) * 255
    return vectorscopeImg.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cb, cr, colors = extractCbCrData(rgbImg)
    vectorscopeImg = createVectorscopeImg(cb, cr, colors)
    cv2.imshow('img', vectorscopeImg)
    cv2.waitKey(0)
