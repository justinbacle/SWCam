import os
import cv2
import subprocess
import config


def saveTiffImg(image, basePath, i):
    cv2.imwrite(f'{basePath}_{i}.tiff', image * 8, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f"Image saved -> {basePath}_{i}.tiff")
    return f"{basePath}_{i}.tiff"


def saveTiffImgAndConvertDNG(binImage, tiffImagePath, i):
    tiffImgPath = saveTiffImg(binImage, tiffImagePath, i)
    dngImgPath, call = convertTiff2Dng(tiffImgPath)
    if call == 0:
        os.remove(tiffImgPath)


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
    call = subprocess.call([config.MAKEDNG_PATH, tiffImagePath, dngImgPath, "3", "7"])
    return dngImgPath, call


def convertTiff2DngAndClean(tiffImgPath):
    _dngImgPath, call = convertTiff2Dng(tiffImgPath)
    if call == 0:
        os.remove(tiffImgPath)
