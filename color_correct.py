import numpy as np

# Sample data taken from https://www.dxomark.com/Cameras/Sony/A7SIII---Measurements

WB_Scale = {
    "CIE_D50": [
        1.61,  # Rraw
        1,  # Graw
        2.43,  # Braw
    ],
    "CIE_A": [
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
    ]
}


def RGBraw2sRGB(rawRGBImage, rgbConversionMatrix):
    rgb_reshaped = rawRGBImage.reshape((rawRGBImage.shape[0] * rawRGBImage.shape[1], rawRGBImage.shape[2]))
    result = np.dot(rgbConversionMatrix, rgb_reshaped.T).T.reshape(rawRGBImage.shape)
    return result


if __name__ == '__main__':
    # TODO test method and check speed
    # TODO add sensor color matrices
    ...
