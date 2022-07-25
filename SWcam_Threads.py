import sys
import logging
import cv2
import numpy as np
import numba  # noqa F401
from PySide6 import QtGui, QtCore
from harvesters.core import Buffer, TimeoutException

from lib import vectorscope
from lib import color_correct


class FrameGrabber(QtCore.QThread):
    def __init__(self, ia, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()

        self.ia = ia

    imageReady = QtCore.Signal(np.ndarray)
    rawImageReady = QtCore.Signal(Buffer)

    def process(self):
        try:
            if not self.isRunning():
                self.start(QtCore.QThread.HighPriority)
            else:
                self.restart = True
                self.condition.wakeOne()
        except Exception as e:
            raise(e)

    def stop(self):
        try:
            self.mutex.lock()
            self.abort = True
            self.condition.wakeOne()
            self.mutex.unlock()
        except Exception as e:
            logging.error(e)

    def run(self):
        self.mutex.lock()
        while self.ia.is_acquiring():
            try:
                with self.ia.fetch(timeout=1) as buffer:
                    component = buffer.payload.components[0]
                    if component is not None:
                        self.rawImageReady.emit(buffer.payload._buffer.raw_buffer)
                        binImage = component.data.reshape(component.height, component.width)
                        self.imageReady.emit(binImage)
                        buffer.queue()
            except TimeoutException:
                logging.error("Timeout error")
            except Exception as e:
                if "Insufficient amount of announced buffers to start acquisition." in str(e) \
                        and sys.platform == "linux":
                    logging.error(
                        """It looks like usb buffer is restricted. Read here to fix it :
                        https://www.flir.eu/support-center/iis/machine-vision/application-note/understanding-usbfs-on-linux
                        orhttps://www.ximea.com/support/wiki/apis/Linux_USB30_Support#Increase-the-USB-Buffer-Size-in-Linux
                    """)
                else:
                    logging.error(e)
        logging.info("Ending Acquisition")
        self.mutex.unlock()


class HistogramProcess(QtCore.QThread):
    def __init__(self, parent=None):
        super(HistogramProcess, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()

        # self.plotWidget = None
        self.image = None

    dataReady = QtCore.Signal(list)
    finished = QtCore.Signal()
    NUM_BINS = 2**10
    bins = np.linspace(int(0), np.log(2**12), NUM_BINS)

    # def setPlottingWidget(self, plotWidget):
    #     self.plotWidget = plotWidget

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()

    def setImg(self, image):
        self.mutex.lock()
        self.image = image
        self.mutex.unlock()

    @staticmethod
    # @numba.njit()  # ? is much slower with numba ???
    def _prepareHistogramData(image, bins):
        DECIMATE_FACTOR = 1
        yR, x = np.histogram(
            np.log(image[0::2*DECIMATE_FACTOR, :].flatten()[0::2*DECIMATE_FACTOR]), bins=bins)
        yG1, x = np.histogram(
            np.log(image[0::2*DECIMATE_FACTOR, :].flatten()[1::2*DECIMATE_FACTOR]), bins=bins)
        yG2, x = np.histogram(
            np.log(image[1::2*DECIMATE_FACTOR, :].flatten()[0::2*DECIMATE_FACTOR]), bins=bins)
        yB, x = np.histogram(
            np.log(image[1::2*DECIMATE_FACTOR, :].flatten()[1::2]*DECIMATE_FACTOR), bins=bins)
        return x, [yR, yG1, yG2, yB]

    def process(self):
        try:
            if not self.isRunning():
                self.start(QtCore.QThread.LowPriority)
            else:
                self.restart = True
                self.condition.wakeOne()
        except Exception as e:
            raise(e)

    def run(self):
        try:
            np.seterr(divide='ignore')
            if self.image is not None:
                x, [yR, yG1, yG2, yB] = self._prepareHistogramData(self.image, self.bins)
                self.dataReady.emit([x, [yR, yG1, yG2, yB]])
                self.image = None
            else:
                logging.debug(f"{self} got {self.image} image to process")
            self.finished.emit()

        except Exception as e:
            raise(e)


class VectorScopeProcess(QtCore.QThread):
    def __init__(self, parent=None):
        super(VectorScopeProcess, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()

        # self.plotWidget = None
        self.image = None

    dataReady = QtCore.Signal(list)
    finished = QtCore.Signal()

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()

    def setImg(self, image):
        self.mutex.lock()
        self.image = image
        self.mutex.unlock()

    def process(self):
        try:
            if not self.isRunning():
                self.start(QtCore.QThread.LowPriority)
            else:
                self.restart = True
                self.condition.wakeOne()
        except Exception as e:
            raise(e)

    def run(self):
        try:
            if self.image is not None:

                # rgbImg = cv2.cvtColor(self.image, cv2.COLOR_BayerRG2RGB)
                # TODO check properly if image is bayer or rgb
                rgbImg = self.image

                rgbImg = cv2.resize(
                    rgbImg, (int(rgbImg.shape[1] * 0.1), int(rgbImg.shape[0] * 0.1)), interpolation=cv2.INTER_AREA)
                cbData, crData, colors = vectorscope.extractCbCrData(rgbImg)
                pos = np.transpose(np.vstack(
                    (
                        cbData-0.5,
                        crData-0.5,
                        np.zeros_like(cbData)
                    )
                ))
                self.dataReady.emit([pos, colors])
            else:
                logging.debug(f"{self} got {self.image} image to process")
            self.finished.emit()

        except Exception as e:
            raise(e)


class ImageProcess(QtCore.QThread):
    def __init__(self, parent=None, drawMethod: str = None):
        super(ImageProcess, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()
        self.IMAGE_DRAW_METHOD = drawMethod

        self.LUT = None
        self.gain = None
        self.binImage = None
        self.CCM = None
        # self.WB = cv2.xphoto.createSimpleWB()
        # self.WB = cv2.xphoto.createGrayworldWB()

    qImageReady = QtCore.Signal(QtGui.QImage)
    imageReady = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()

    def setLUT(self, LUT):
        self.mutex.lock()
        self.LUT = LUT
        self.mutex.unlock()

    def setGain(self, redGain, greenGain, blueGain):
        self.mutex.lock()
        if self.IMAGE_DRAW_METHOD == "QtImageViewer":
            # reverse order for some reason
            self.gain = [blueGain, greenGain, redGain]
        else:
            self.gain = [redGain, greenGain, blueGain]
        self.mutex.unlock()

    def setImg(self, binImage):
        self.mutex.lock()
        self.binImage = binImage
        self.mutex.unlock()

    def setCCM(self, CCM: np.ndarray):
        self.mutex.lock()
        self.CCM = CCM
        self.mutex.unlock()

    def process(self):
        try:
            if not self.isRunning():
                self.start(QtCore.QThread.NormalPriority)
            else:
                self.restart = True
                self.condition.wakeOne()
        except Exception as e:
            raise e

    @staticmethod
    def _processImg(rgbImg, LUT=None, CLAHE=None, colorMatrix=None, Gain: list = None, Gamma: float = None, WB=None):

        rgbImg = cv2.cvtColor(np.uint16(rgbImg), cv2.COLOR_BAYER_RG2BGR_EA)  # should be RGB instead of BGR

        # CLAHE METHOD
        if CLAHE is not None:
            for i in range(3):
                rgbImg[i] = CLAHE.apply(rgbImg[i])
            rgbImg = cv2.convertScaleAbs(rgbImg)

        if colorMatrix is not None:
            rgbImg = color_correct.RGBraw2sRGB(rgbImg, colorMatrix)

        if Gain is not None:
            Gain = np.array(Gain) / max(Gain)
            for i, colorGain in enumerate(Gain):
                rgbImg[:, :, i] = rgbImg[:, :, i] * colorGain

        # 8-Bit LUT METHOD FAST, noisy  <-- works
        if LUT is not None:
            rgbImg = cv2.convertScaleAbs(rgbImg, alpha=1/8)
            rgbImg = cv2.LUT(rgbImg, LUT)

        if Gamma is not None:
            # Gamma Luma METHOD NOT WORKING
            # rgbImg = rgbImg.astype(np.float32)
            # labImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2Lab)
            # mid = 0.5
            # mean = np.mean(labImg[0])
            # gamma = math.log(mid*255)/math.log(mean)
            # labImg[0] = np.power(labImg[0], gamma)
            # rgbImg = cv2.cvtColor(labImg, cv2.COLOR_Lab2RGB)
            # rgbImg = cv2.convertScaleAbs(rgbImg).astype(np.uint8)

            # # Gamma RGB METHOD SLOW
            rgbImg = rgbImg.astype(np.float32) / np.max(rgbImg)
            rgbImg = np.power(rgbImg, Gamma)
            rgbImg = rgbImg * 255
            rgbImg = cv2.convertScaleAbs(rgbImg).astype(np.uint8)

        # Auto White Balance
        if WB is not None:
            rgbImg = WB.balanceWhite(rgbImg)

        return rgbImg.astype(np.uint8)

    def run(self):
        try:
            if self.binImage is not None:
                # rgbImg = processImg(
                #     self.binImage, Gamma=None, LUT=self.LUT, Gain=self.gain)
                rgbImg = self._processImg(
                    self.binImage,
                    # colorMatrix=color_correct.COLOR_MATRIX["BFLY-U3-23S6C"],
                    colorMatrix=self.CCM,
                    LUT=self.LUT,
                    # Gain=color_correct.WB_Scale["CIE-D50"],
                    Gain=self.gain,
                )
                self.imageReady.emit(rgbImg)
                # qImg = QtGui.QImage(
                #     rgbImg.data,
                #     rgbImg.shape[1],
                #     rgbImg.shape[0],
                #     rgbImg.shape[1] * rgbImg.shape[2],
                #     QtGui.QImage.Format_RGB888
                # )
                # self.qImageReady.emit(qImg)
            else:
                logging.debug(f"{self} got {self.binImage} image to process")
            self.finished.emit()
        except Exception as e:
            raise(e)
