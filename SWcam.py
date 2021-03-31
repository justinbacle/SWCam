import sys
import os
import time
from PySide6 import QtWidgets, QtGui, QtOpenGL, QtCore  # noqa F401
from harvesters.core import Harvester
from harvesters.core import TimeoutException
import logging
import cv2
import numpy as np
import math  # noqa F401
import datetime
import pathlib
import threading
import tqdm
import pyqtgraph

import utils
import config
import imageIO


class FrameGrabber(QtCore.QObject):
    def __init__(self, ia, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.ia = ia

    imageReady = QtCore.Signal(np.ndarray)

    def run(self):
        while self.ia.is_acquiring():
            try:
                with self.ia.fetch_buffer(timeout=1) as buffer:
                    component = buffer.payload.components[0]
                    if component is not None:
                        binImage = component.data.reshape(component.height, component.width)
                        self.imageReady.emit(binImage)
            except TimeoutException:
                logging.error(f"Timeout error")


class HistogramProcess(QtCore.QObject):
    def __init__(self, parent=None):
        super(HistogramProcess, self).__init__(parent)
        # self.plotWidget = None
        self.image = None

    dataReady = QtCore.Signal(list)
    NUM_BINS = 64

    # def setPlottingWidget(self, plotWidget):
    #     self.plotWidget = plotWidget

    def setImg(self, image):
        self.image = image

    def run(self):
        while True:
            if self.image is not None:
                yR, x = np.histogram(
                    np.log(self.image[0::2, :].flatten()[0::2]),
                    bins=np.linspace(0, np.log(2**12), num=self.NUM_BINS)
                )
                yG1, x = np.histogram(
                    np.log(self.image[0::2, :].flatten()[1::2]),
                    bins=np.linspace(0, np.log(2**12), num=self.NUM_BINS)
                )
                yG2, x = np.histogram(
                    np.log(self.image[1::2, :].flatten()[0::2]),
                    bins=np.linspace(0, np.log(2**12), num=self.NUM_BINS)
                )
                yB, x = np.histogram(
                    np.log(self.image[1::2, :].flatten()[1::2]),
                    bins=np.linspace(0, np.log(2**12), num=self.NUM_BINS)
                )
                self.dataReady.emit([x, [yR, yG1, yG2, yR]])
                self.image = None
            else:
                time.sleep(0.01)


class ImageProcess(QtCore.QObject):
    def __init__(self, parent=None):
        super(ImageProcess, self).__init__(parent)
        self.LUT = None
        self.binImage = None
        # self.WB = cv2.xphoto.createSimpleWB()
        # self.WB = cv2.xphoto.createGrayworldWB()

    qImageReady = QtCore.Signal(QtGui.QImage)
    imageReady = QtCore.Signal(np.ndarray)

    def setLUT(self, LUT):
        self.LUT = LUT

    def setImg(self, binImage):
        self.binImage = binImage

    def run(self):
        while True:
            if self.binImage is not None:
                rgbImg = processImg(self.binImage, LUT=self.LUT)
                self.imageReady.emit(rgbImg)
                qImg = QtGui.QImage(
                    rgbImg.data,
                    rgbImg.shape[1],
                    rgbImg.shape[0],
                    rgbImg.shape[1] * rgbImg.shape[2],
                    QtGui.QImage.Format_RGB888
                )
                self.qImageReady.emit(qImg)
                self.binImage = None
            else:
                time.sleep(0.01)


def processImg(rgbImg, LUT=None, CLAHE=None, Gamma=None, WB=None):

    # CLAHE METHOD
    if CLAHE is not None:
        rgbImg = cv2.cvtColor(np.uint16(rgbImg), cv2.COLOR_BAYER_RG2RGB_EA)
        for i in range(3):
            rgbImg[i] = CLAHE.apply(rgbImg[i])
        rgbImg = cv2.convertScaleAbs(rgbImg)

    # 8-Bit LUT METHOD FAST, noisy  <-- works
    elif LUT is not None:
        rgbImg = cv2.cvtColor(np.uint16(rgbImg), cv2.COLOR_BAYER_RG2RGB_EA)
        rgbImg = cv2.convertScaleAbs(rgbImg, alpha=1/8)
        rgbImg = cv2.LUT(rgbImg, LUT)

    # Gamma Luma METHOD NOT WORKING
    elif Gamma is not None:
        rgbImg = cv2.cvtColor(np.uint16(rgbImg), cv2.COLOR_BAYER_RG2RGB_EA)
        rgbImg = rgbImg.astype(np.float32)
        labImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2Lab)
        mid = 0.5
        mean = np.mean(labImg[0])
        gamma = math.log(mid*255)/math.log(mean)
        labImg[0] = np.power(labImg[0], gamma)
        rgbImg = cv2.cvtColor(labImg, cv2.COLOR_Lab2RGB)
        rgbImg = cv2.convertScaleAbs(rgbImg).astype(np.uint8)

        # # Gamma RGB METHOD SLOW
        # rgbImg = cv2.cvtColor(np.uint16(binImage), cv2.COLOR_BAYER_RG2RGB_EA)
        # mid = 0.5
        # mean = np.mean(rgbImg)
        # gamma = math.log(mid*255)/math.log(mean)
        # for i in range(3):
        #     rgbImg[i] = np.power(rgbImg[i], gamma)
        # rgbImg = cv2.convertScaleAbs(rgbImg)

    # Auto White Balance
    if WB is not None:
        rgbImg = WB.balanceWhite(rgbImg)

    return rgbImg


# LUT GENERATOR
def processPixelLUT(x):
    return (np.log(1 + x + 20) - 3.044522437723423) / 5.545177444479562 * 2**8 / 180 * 255


class SWCameraGui(QtWidgets.QWidget):
    def __init__(self):
        super(SWCameraGui, self).__init__()
        self.initUI()
        self.initCam()
        self.initConstants()
        # self.showMaximized()
        # self.initThreads()
        self.initProcessing()
        self.show()

    # ------ INIT ------

    def initConstants(self):
        self.OUTPUT_PATH = "D:/VideoOut"
        self.save_threads = []
        self.imageCount = utils.counter()
        self.savePath = None

    def initAcquisitionThread(self):
        # Acquisition Thread
        self.acquisitionThread = QtCore.QThread()
        self.grabber = FrameGrabber(self.ia)
        self.grabber.moveToThread(self.acquisitionThread)
        self.grabber.imageReady.connect(self.clbkProcessImage)
        self.grabber.imageReady.connect(self.saveImgThread)
        self.grabber.imageReady.connect(self.updateHistogram)
        self.acquisitionThread.started.connect(self.grabber.run)
        self.acquisitionThread.start()

    def initImageProcessThread(self):
        self.imageProcessThread = QtCore.QThread()
        self.imageProcess = ImageProcess()
        self.imageProcess.moveToThread(self.imageProcessThread)
        self.imageProcess.qImageReady.connect(self.drawQimg)
        # self.imageProcess.imageReady.connect(self.updateHistogram)
        self.imageProcessThread.started.connect(self.imageProcess.run)

        self.histogramProcessThread = QtCore.QThread()
        self.histogramProcess = HistogramProcess()
        self.histogramProcess.moveToThread(self.histogramProcessThread)
        # self.histogramProcess.setPlottingWidget(self.histographPlot)
        self.histogramProcess.dataReady.connect(self.updateHistogramWidget)
        self.histogramProcessThread.started.connect(self.histogramProcess.run)

    def initUI(self):
        # main layout
        self.mainLayout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.mainLayout)

        # image preview
        self.imageViewer = utils.QtImageViewer()
        self.imageViewer.setMinimumSize(960, 600)
        self.mainLayout.addWidget(self.imageViewer)

        # Right Panel

        self.rightLayout = QtWidgets.QVBoxLayout()

        i = utils.counter()
        self.controlLayout = QtWidgets.QGridLayout()
        self.connectButton = QtWidgets.QPushButton("Connect")
        self.connectButton.pressed.connect(self.clbkConnectCamera)
        self.controlLayout.addWidget(self.connectButton, i.postinc(), 0, 1, 2)
        self.runButton = QtWidgets.QPushButton("Run")
        self.runButton.pressed.connect(self.clbkRunCamera)
        self.runButton.setEnabled(False)
        self.controlLayout.addWidget(self.runButton, i.postinc(), 0, 1, 2)
        self.recordButton = QtWidgets.QPushButton("Record")
        self.recordButton.setCheckable(True)
        self.recordButton.setEnabled(False)
        self.recordButton.setStyleSheet("QPushButton:checked {background-color: red;}")  # Red means recording
        self.recordButton.toggled.connect(self.clbkRecord)
        self.controlLayout.addWidget(self.recordButton, i.postinc(), 0, 1, 2)

        _horizontalResolutionLabel = QtWidgets.QLabel("Horizontal Resolution :")
        self.controlLayout.addWidget(_horizontalResolutionLabel, i.val(), 0)
        self.horizontalResolution = QtWidgets.QSpinBox()
        self.horizontalResolution.setSingleStep(4)
        self.horizontalResolution.valueChanged.connect(self.clbkResolutionChange)
        self.controlLayout.addWidget(self.horizontalResolution, i.postinc(), 1)
        _verticalResolutionLabel = QtWidgets.QLabel("vertical Resolution :")
        self.controlLayout.addWidget(_verticalResolutionLabel, i.val(), 0)
        self.verticalResolution = QtWidgets.QSpinBox()
        self.verticalResolution.setSingleStep(4)
        self.verticalResolution.valueChanged.connect(self.clbkResolutionChange)
        self.controlLayout.addWidget(self.verticalResolution, i.postinc(), 1)
        self.initResolutionSpinBoxes()

        _framerateLabel = QtWidgets.QLabel("Framerate :")
        self.controlLayout.addWidget(_framerateLabel, i.val(), 0)
        self.framerate = QtWidgets.QDoubleSpinBox()
        self.framerate.valueChanged.connect(self.clbkFramerateChange)
        self.controlLayout.addWidget(self.framerate, i.postinc(), 1)

        _gainLabel = QtWidgets.QLabel("Gain (dB) :")
        self.controlLayout.addWidget(_gainLabel, i.val(), 0)
        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.valueChanged.connect(self.clbkGainChange)
        self.controlLayout.addWidget(self.gain, i.postinc(), 1)

        _shutterLabel = QtWidgets.QLabel("shutter (°) :")
        self.controlLayout.addWidget(_shutterLabel, i.val(), 0)
        self.shutter = QtWidgets.QDoubleSpinBox()
        self.shutter.valueChanged.connect(self.clbkShutterChange)
        self.shutter.setMaximum(360.0)
        self.controlLayout.addWidget(self.shutter, i.postinc(), 1)

        _modeLabel = QtWidgets.QLabel("Video Mode :")
        self.controlLayout.addWidget(_modeLabel, i.val(), 0)
        self.mode = QtWidgets.QComboBox()
        self.mode.currentTextChanged.connect(self.clbkModeChange)
        self.controlLayout.addWidget(self.mode, i.postinc(), 1)

        self.rightLayout.addLayout(self.controlLayout)

        self.histogram = pyqtgraph.GraphicsLayoutWidget()
        self.histographPlot = self.histogram.addPlot()
        self.histogram.setMaximumSize(320, 120)
        # self.histographPlot.setLogMode(False, True)
        self.rightLayout.addWidget(self.histogram)

        self.mainLayout.addLayout(self.rightLayout)

    def initResolutionSpinBoxes(self):
        self.horizontalResolution.setMaximum(1920)
        self.verticalResolution.setMaximum(1200)

    def initCam(self):
        self.harvester = Harvester()
        self.harvester.add_file(config.CTI_FILEPATH)
        self.harvester.update()
        # TODO handle multiple cameras with QComboBox ?
        try:
            self.ia = self.harvester.create_image_acquirer(0)
        except IndexError:
            self.ia = None
            logging.error("No Camera Available")

    def initProcessing(self):
        curve = np.arange(2**8, dtype=np.dtype('uint8'))
        curve = np.multiply(np.uint16(curve), 4)
        curve = processPixelLUT(curve).astype('uint8')
        self.LUT = np.dstack((curve, curve, curve))

        self.CLAHE = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    # ------ Image processing ------

    def drawImg(self, image: np.ndarray):
        qImg = QtGui.QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.shape[1] * image.shape[2],
            QtGui.QImage.Format_RGB888
        )
        self.imageViewer.setImage(qImg.rgbSwapped())

    def drawQimg(self, qImg: QtGui.QImage):
        self.imageViewer.setImage(qImg.rgbSwapped())

    def clbkProcessImage(self, binImage):
        self.imageProcess.setImg(binImage)
        self.imageProcessThread.start()

    def updateHistogram(self, rgbImage):
        # too slow for real time ? put in thread ?
        self.histogramProcess.setImg(rgbImage)
        self.histogramProcessThread.start()

    def updateHistogramWidget(self, dataList):
        x, y = dataList
        yR, yG1, yG2, yB = y
        # self.histographPlot.plot(x, y, stepMode=True, clear=True)
        self.histographPlot.plot(x, yR, stepMode=True, pen=(255, 0, 0), clear=True)
        self.histographPlot.plot(x, yG1, stepMode=True, pen=(0, 255, 0))
        self.histographPlot.plot(x, yG2, stepMode=True, pen=(0, 255, 0))
        self.histographPlot.plot(x, yB, stepMode=True, pen=(0, 0, 255))

    # ------ CALBACKS ------

    def clbkConnectCamera(self):
        # Init camera parameters
        if self.ia is not None:
            self.ia.remote_device.node_map.DeviceLinkThroughputLimit.value = \
                self.ia.remote_device.node_map.DeviceMaxThroughput.value
            self.ia.remote_device.node_map.AcquisitionFrameRateAuto.value = 'Off'
            self.ia.remote_device.node_map.AcquisitionFrameRateEnabled.value = True
            self.ia.remote_device.node_map.BalanceWhiteAuto.value = 'Off'
            self.ia.remote_device.node_map.pgrExposureCompensation.value = float(0.0)
            # TODO have selector for that
            self.ia.remote_device.node_map.PixelFormat.value = 'BayerRG12p'

            self.ia.remote_device.node_map.BalanceRatioSelector.value = 'Red'
            self.ia.remote_device.node_map.BalanceRatio.value = float(1.0)  # how to Kelvin ?
            self.ia.remote_device.node_map.BalanceRatioSelector.value = 'Blue'
            self.ia.remote_device.node_map.BalanceRatio.value = float(1.0)  # how to Kelvin ?
            self.ia.remote_device.node_map.BlackLevel.value = float(0.0)
            self.ia.remote_device.node_map.GainAuto.value = 'Off'

            self.clbkReadCameraParams()
            self.clbkUpdateVideoModes()
            self.clbkUpdateUiLimits()
            self.runButton.setEnabled(True)
        else:
            logging.error("No camera is available")

    def clbkReadCameraParams(self):
        self.horizontalResolution.setValue(self.ia.remote_device.node_map.Width.value)
        self.verticalResolution.setValue(self.ia.remote_device.node_map.Height.value)
        self.framerate.setValue(self.ia.remote_device.node_map.AcquisitionFrameRate.value)
        self.gain.setValue(self.ia.remote_device.node_map.Gain.value)
        self.shutter.setValue(
            self.ia.remote_device.node_map.ExposureTime.value / 1e6 / (1 / self.framerate.value()) * 360)
        # TODO for video mode

    def clbkUpdateUiLimits(self):
        self.horizontalResolution.setMaximum(self.ia.remote_device.node_map.WidthMax.value)
        self.verticalResolution.setMaximum(self.ia.remote_device.node_map.HeightMax.value)

    def clbkRunCamera(self):
        if self.ia.is_acquiring():
            self.ia.stop_acquisition()
            self.runButton.setText("Run")
            self.recordButton.setEnabled(False)
        else:
            self.ia.start_acquisition()
            self.initAcquisitionThread()
            self.acquisitionThread.start()
            self.initImageProcessThread()
            self.imageProcess.setLUT(self.LUT)
            self.imageProcessThread.start()
            self.runButton.setText("Stop")
            self.recordButton.setEnabled(True)
            self.histogramProcessThread.start()

    def saveImgThread(self, binImage):
        if self.recordButton.isChecked():
            # TODO add radio buttons for formats + path
            if self.savePath is not None:
                self.save_threads.append(
                    threading.Thread(
                        target=imageIO.saveTiffImg,
                        args=(binImage, f"{self.savePath}/IMG", self.imageCount.postinc())
                    )
                )
                self.save_threads[-1].start()
            else:
                print('no savePath ?')

    def clbkRecord(self):
        if self.recordButton.isChecked():
            self.saveFolder = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.savePath = pathlib.Path(self.OUTPUT_PATH, self.saveFolder)
            os.mkdir(self.savePath)
            self.imageCount.set(0)
            logging.info(f"Started recording in {self.saveFolder}")
        else:
            # clear self.save_threads = [] when done
            self.deferredTiffToDng(self.savePath)

    def deferredTiffToDng(self, savePath):
        save_threads = []
        files = os.listdir(savePath)
        for file in tqdm.tqdm(files):
            save_threads.append(
                threading.Thread(
                    target=imageIO.convertTiff2DngAndClean,
                    args=(str(pathlib.Path(savePath, file)),)
                )
            )
            save_threads[-1].start()

    def clbkResolutionChange(self):
        try:
            self.ia.remote_device.node_map.OffsetX.value = int(0)
            self.ia.remote_device.node_map.OffsetY.value = int(0)
            self.ia.remote_device.node_map.Width.value = int(self.horizontalResolution.text())
            self.ia.remote_device.node_map.Height.value = int(self.verticalResolution.text())
            self.ia.remote_device.node_map.OffsetX.value = int(
                (self.ia.remote_device.node_map.WidthMax.value - int(self.horizontalResolution.text())) / 2)
            self.ia.remote_device.node_map.OffsetY.value = int(
                (self.ia.remote_device.node_map.HeightMax.value - int(self.verticalResolution.text())) / 2)
        except Exception as e:
            print(e)
        self.horizontalResolution.setValue(self.ia.remote_device.node_map.Width.value)
        self.verticalResolution.setValue(self.ia.remote_device.node_map.Height.value)

    def clbkShutterChange(self):
        try:
            self.ia.remote_device.node_map.ExposureTime.value = \
                1 / self.framerate.value() * self.shutter.value()/360 * 1e6
        except:  # noqa E722
            print(f"could not set shutter to {self.shutter.value()}")
        self.shutter.setValue(
            self.ia.remote_device.node_map.ExposureTime.value / 1e6 / (1 / self.framerate.value()) * 360)

    def clbkFramerateChange(self):
        try:
            self.ia.remote_device.node_map.AcquisitionFrameRate.value = self.framerate.value()
        except:  # noqa E722
            print(f"could not set framerate to {self.framerate.value()}")
        self.framerate.setValue(self.ia.remote_device.node_map.AcquisitionFrameRate.value)
        self.clbkShutterChange()

    def clbkGainChange(self):
        try:
            self.ia.remote_device.node_map.Gain.value = self.gain.value()
        except:  # noqa E722
            print(f"could not set gain to {self.gain.value()}")
        self.gain.setValue(self.ia.remote_device.node_map.Gain.value)

    def clbkModeChange(self):
        try:
            self.ia.remote_device.node_map.VideoMode.value = self.mode.currentText()
        except:  # noqa E722
            print(f"could not set mode to {self.mode.currentText()}")

    def clbkUpdateVideoModes(self):
        self.mode.clear()
        for mode in self.ia.remote_device.node_map.VideoMode.symbolics:
            self.mode.addItem(mode)


def SWCamera():
    app = QtWidgets.QApplication(sys.argv)
    _gui = SWCameraGui()  # noqa F841

    sys.exit(app.exec_())


if __name__ == '__main__':
    SWCamera()
