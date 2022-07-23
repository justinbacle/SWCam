import os
import cv2
import subprocess
import config
import sys


def saveTiffImg(image, basePath, i):
    cv2.imwrite(f'{basePath}_{i}.tiff', image * 8, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f"Image saved -> {basePath}_{i}.tiff")
    return f"{basePath}_{i}.tiff"


def saveTiffImgAndConvertDNG(binImage, tiffImagePath, i):
    tiffImgPath = saveTiffImg(binImage, tiffImagePath, i)
    dngImgPath, call = convertTiff2Dng(tiffImgPath)
    if call == 0:
        os.remove(tiffImgPath)


def saveRawImg(rawData, basePath, i):
    with open(f'{basePath}_{i}.raw', 'wb') as f_:
        f_.write(rawData)
    print(f"Image saved -> {basePath}_{i}.raw")
    return f"{basePath}_{i}.raw"


def convertTiff2Dng(tiffImagePath):
    dngImgPath = tiffImagePath.replace('.tiff', '.dng')
    """usage: makeDNG input_tiff_file output_dng_file [cfa_pattern] [compression]
               [reelname] [frame number]

       cfa_pattern 0: BGGR
                   1: GBRG
                   2: GRBG
                   3: RGGB (default)

       compression 1: none (default) -> 4509 kB
                   7: lossless JPEG -> 2335 kB
                   8: Adobe Deflate (16-bit float) -> Slow / ~2000 kB
    """
    CFA_PATTERN_BGGR = "0"  # noqa F841
    CFA_PATTERN_GBRG = "1"  # noqa F841
    CFA_PATTERN_GRBG = "2"  # noqa F841
    CFA_PATTERN_RGGB = "3"  # noqa F841

    COMPRESSION_NONE = "1"  # noqa F841
    COMPRESSION_LJPG = "7"  # noqa F841
    COMPRESSION_ADEF = "8"  # noqa F841

    if sys.platform == "linux":
        # Wine call, temporary fix for linux
        call = subprocess.call(
            ["wine", config.MAKEDNG_PATH, tiffImagePath, dngImgPath, CFA_PATTERN_RGGB, COMPRESSION_LJPG])
    else:
        call = subprocess.call(
            [config.MAKEDNG_PATH, tiffImagePath, dngImgPath, CFA_PATTERN_RGGB, COMPRESSION_LJPG])
    return dngImgPath, call


def convertTiff2DngAndClean(tiffImgPath):
    _dngImgPath, call = convertTiff2Dng(tiffImgPath)
    if call == 0:
        os.remove(tiffImgPath)
