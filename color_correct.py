import numpy as np
import sys
import cv2
import numba

# Sample data taken from https://www.dxomark.com/Cameras/Sony/A7SIII---Measurements

WB_Scale = {
    "CIE-D50": [
        1.61,  # Rraw
        1,  # Graw
        2.43,  # Braw
    ],
    "CIE-A": [
        2.44,  # Rraw
        1,  # Graw
        1.51,  # Braw
    ]
}

# WB for BF-U3... https://github.com/ifb/makeDNG/blob/master/makeDNG.c

COLOR_MATRIX = {
    "CIE-D50": [
        # RsRGB, GsRGB, BsRGB
        [1.9, -0.71, -0.19],  # Rraw
        [-0.17, 1.47, -0.3],  # Graw
        [0.13, -0.85, 1.72],  # Braw
    ],
    "CIE-A": [
        [2.31, -1.23, -0.08],
        [-0.18, 1.65, -0.47],
        [0, -0.51, 1.51],
    ],
    "BFLY-U3-23S6C": [
        [1.8, -0.25, -0.5],
        [-0.35, 1.2, -0.2],
        [-0.15, -0.15, 2.3],
    ],
}

TEST_COLOR_MATRIX = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
]


@numba.njit()
def applyColorMatrix(floatImgData, rgbConversionMatrix):
    """ From https://stackoverflow.com/questions/22081423/apply-transformation-matrix-to-pixels-in-opencv-image """
    rgb_reshaped = floatImgData.reshape((floatImgData.shape[0] * floatImgData.shape[1], floatImgData.shape[2]))
    result = np.dot(rgbConversionMatrix, rgb_reshaped.T).T
    return result.reshape(floatImgData.shape)


def RGBraw2sRGB(rawRGBImage, rgbConversionMatrix, avoidClipping: bool = True):
    # convert to float type for conversion
    dataType = rawRGBImage.dtype
    if dataType == np.uint8:
        rawRGBImage = rawRGBImage.astype(float) / 2**8
    elif dataType == np.uint16:
        rawRGBImage = rawRGBImage.astype(float) / 2**16
    elif dataType == float:
        ...
    else:
        print(f"dataType {dataType} is unsupported")

    result = applyColorMatrix(rawRGBImage, np.array(rgbConversionMatrix, dtype=float))

    if avoidClipping:
        maxImgValue = np.max(result)
        if maxImgValue > 1:
            result = result / maxImgValue

    # convert back to initial type
    if dataType == np.uint8:
        result = (result*2**8).astype(np.uint8)
    elif dataType == np.uint16:
        result = (result*2**16).astype(np.uint16)
    elif dataType == float:
        ...

    return result


if __name__ == '__main__':
    # TODO test method and check speed
    # TODO add sensor color matrices
    if sys.platform == "win32":
        img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    elif sys.platform == "linux":
        img = cv2.imread("/home/jjj/Pictures/MIN-2K4-KNA_A_FBL.jpg")

    img = cv2.resize(
        img,
        (
            int(img.shape[0] * 0.25),
            int(img.shape[1] * 0.25)
        )
    )
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', rgbImg)
    rgbCImg = RGBraw2sRGB(rgbImg, COLOR_MATRIX["CIE-D50"])
    # rgbCImg = RGBraw2sRGB(rgbImg, TEST_COLOR_MATRIX)
    cv2.imshow('imgC', rgbCImg)
    cv2.waitKey(0)
