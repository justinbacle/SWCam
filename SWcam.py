import sys
import os
from PySide2 import QtWidgets, QtGui, QtCore
from harvesters.core import Harvester
from harvesters.core import TimeoutException
import logging
import cv2
import numpy as np
import datetime
import pathlib
import threading
import tqdm
import pyqtgraph
import vispy
import vispy.scene

import utils
import config
import imageIO
import vectorscope
import color_correct


HISTOGRAM = False
VECTORSCOPE = True

# IMAGE_DRAW_METHOD = "QtImageViewer"
IMAGE_DRAW_METHOD = "Vispy"


class FrameGrabber(QtCore.QThread):
    def __init__(self, ia, parent=None):
        super(FrameGrabber, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()

        self.ia = ia

    imageReady = QtCore.Signal(np.ndarray)

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
            print(e)

    def run(self):
        self.mutex.lock()
        while self.ia.is_acquiring():
            try:
                with self.ia.fetch_buffer(timeout=1) as buffer:
                    component = buffer.payload.components[0]
                    if component is not None:
                        binImage = component.data.reshape(component.height, component.width)
                        self.imageReady.emit(binImage)
            except TimeoutException:
                logging.error("Timeout error")
            except Exception as e:
                print(e)
        print("Ending Acquisition")
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
                x, [yR, yG1, yG2, yR] = prepareHistogramData(self.image, self.bins)
                self.dataReady.emit([x, [yR, yG1, yG2, yR]])
                self.image = None
            else:
                print(f"{self} got {self.image} image to process")
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
                self.start(QtCore.QThread.NormalPriority)
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
                print(f"{self} got {self.image} image to process")
            self.finished.emit()

        except Exception as e:
            raise(e)


class ImageProcess(QtCore.QThread):
    def __init__(self, parent=None):
        super(ImageProcess, self).__init__(parent)
        self.mutex = QtCore.QMutex()
        self.condition = QtCore.QWaitCondition()

        self.LUT = None
        self.gain = None
        self.binImage = None
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

    def setGamma(self, redGain, greenGain, blueGain):
        self.mutex.lock()
        if IMAGE_DRAW_METHOD == "QtImageViewer":
            # reverse order for some reason
            self.gain = [blueGain, greenGain, redGain]
        else:
            self.gain = [redGain, greenGain, blueGain]
        self.mutex.unlock()

    def setImg(self, binImage):
        self.mutex.lock()
        self.binImage = binImage
        self.mutex.unlock()

    def process(self):
        try:
            if not self.isRunning():
                self.start(QtCore.QThread.LowPriority)
            else:
                self.restart = True
                self.condition.wakeOne()
        except Exception as e:
            raise e

    def run(self):
        try:
            if self.binImage is not None:
                # rgbImg = processImg(
                #     self.binImage, Gamma=None, LUT=self.LUT, Gain=self.gain)
                rgbImg = processImg(
                    self.binImage,
                    colorMatrix=color_correct.COLOR_MATRIX["BFLY-U3-23S6C"],
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
                print(f"{self} got {self.binImage} image to process")
            self.finished.emit()
        except Exception as e:
            raise(e)


def prepareHistogramData(image, bins):
    yR, x = np.histogram(
        np.log(image[0::2, :].flatten()[0::2]),
        bins=bins
    )
    yG1, x = np.histogram(
        np.log(image[0::2, :].flatten()[1::2]),
        bins=bins
    )
    yG2, x = np.histogram(
        np.log(image[1::2, :].flatten()[0::2]),
        bins=bins
    )
    yB, x = np.histogram(
        np.log(image[1::2, :].flatten()[1::2]),
        bins=bins
    )
    return x, [yR, yG1, yG2, yR]


def processImg(rgbImg, LUT=None, CLAHE=None, colorMatrix=None, Gain: list = None, Gamma: float = None, WB=None):

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
        self.initProcessing()
        self.show()

    # ------ INIT ------

    def initConstants(self):
        if sys.platform == "win32":
            self.OUTPUT_PATH = "D:/VideoOut"
        elif sys.platform == "linux":
            self.OUTPUT_PATH = "/media/jjj/7B2F-AED5/VideoOut"
        self.save_threads = []
        self.imageCount = utils.counter()
        self.savePath = None

    def initAcquisitionThread(self):
        # Acquisition Thread
        self.grabber = FrameGrabber(self.ia)
        self.grabber.imageReady.connect(self.saveImgThread)
        self.grabber.imageReady.connect(self.clbkProcessImage)
        if HISTOGRAM:
            self.grabber.imageReady.connect(self.updateHistogram)
        # if VECTORSCOPE:
        #     self.grabber.imageReady.connect(self.updateVectorScope)

    def initImageProcessThread(self):
        self.imageProcess = ImageProcess()
        self.imageProcess.imageReady.connect(self.drawImg)
        if VECTORSCOPE:
            self.imageProcess.imageReady.connect(self.updateVectorScope)

        if HISTOGRAM:
            self.histogramProcess = HistogramProcess()
            self.histogramProcess.dataReady.connect(self.updateHistogramWidget)
        if VECTORSCOPE:
            self.vectorScopeProcess = VectorScopeProcess()
            self.vectorScopeProcess.dataReady.connect(self.updateVectorScopeWidget)

    def initUI(self):
        # main layout
        self.mainLayout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.mainLayout)

        # image preview
        if IMAGE_DRAW_METHOD == "QtImageViewer":
            self.imageViewer = utils.QtImageViewer()
            self.imageViewer.setMinimumSize(960, 600)
            self.mainLayout.addWidget(self.imageViewer)
        elif IMAGE_DRAW_METHOD == "Vispy":
            self.imageViewerCanvas = vispy.scene.SceneCanvas(title="Preview", show=True)
            self.imageViewerCanvas.show()
            self.imageViewer = self.imageViewerCanvas.central_widget.add_view()
            self.imageViewerPhoto = vispy.scene.visuals.Image(
                data=np.ndarray((1200, 1920, 3), dtype=np.uint8),  # TODO replace with some wisely chosen constant
                parent=self.imageViewer.scene,
                method='auto',
                # interpolation="catrom",
                interpolation="bilinear",
            )
            self.imageViewer.camera = 'panzoom'
            self.imageViewer.camera.flip = (False, True)
            self.imageViewer.camera.aspect = 1.0
            self.imageViewer.camera.rotation = 90
            # self.imageViewer.camera.set_range((0, 800), (0, 600))
            self.imageViewer.camera.set_range()
            self.imageViewer.camera.fov = 0
            self.mainLayout.addWidget(self.imageViewerCanvas.native)

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

        # RGB controls (preview)
        _rgbLabel = QtWidgets.QLabel("RGB Preview Controls :")
        self.controlLayout.addWidget(_rgbLabel, i.postinc(), 0)
        self.redGain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.redGain.setMinimum(0)
        self.redGain.setMaximum(100)
        self.redGain.setValue(100)
        self.redGain.valueChanged.connect(self.updateRGBGain)
        self.controlLayout.addWidget(self.redGain, i.val(), 1)
        self.redGammaLabel = QtWidgets.QLabel()
        self.controlLayout.addWidget(self.redGammaLabel, i.postinc(), 0)
        self.greenGamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.greenGamma.setMinimum(0)
        self.greenGamma.setMaximum(100)
        self.greenGamma.setValue(100)
        self.greenGamma.valueChanged.connect(self.updateRGBGain)
        self.controlLayout.addWidget(self.greenGamma, i.val(), 1)
        self.greenGammaLabel = QtWidgets.QLabel()
        self.controlLayout.addWidget(self.greenGammaLabel, i.postinc(), 0)
        self.blueGain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.blueGain.setMinimum(0)
        self.blueGain.setMaximum(100)
        self.blueGain.setValue(100)
        self.blueGain.valueChanged.connect(self.updateRGBGain)
        self.controlLayout.addWidget(self.blueGain, i.val(), 1)
        self.blueGammaLabel = QtWidgets.QLabel()
        self.controlLayout.addWidget(self.blueGammaLabel, i.postinc(), 0)

        self.rightLayout.addLayout(self.controlLayout)

        # Histogram
        if HISTOGRAM:
            self.histogram = pyqtgraph.GraphicsLayoutWidget()
            self.histographPlot = self.histogram.addPlot()
            # self.histogram.setMaximumSize(320, 120)
            # self.histographPlot.setLogMode(False, True)
            self.rightLayout.addWidget(self.histogram)

        # vispy Vectorscope
        if VECTORSCOPE:
            self.vectorScopeCanvas = vispy.scene.SceneCanvas(size=(400, 400))
            self.rightLayout.addWidget(self.vectorScopeCanvas.native)
            self.initVectorScopeWidget()

        self.mainLayout.addLayout(self.rightLayout)

    def initVectorScopeWidget(self):
        Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
        view = self.vectorScopeCanvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 0
        view.camera.set_range(
            x=(-5, 5),
            y=(-5, 5),
            margin=0.0
        )
        view.camera.interactive = False
        self.vectorScopePlot = Scatter3D(parent=view.scene)
        self.vectorScopePlot.set_gl_state('translucent', blend=True, depth_test=True)
        self.vectorScopePlot.antialias = 0

        # TODO add overlay
        # vispy.scene.visuals.Line(
        #     pos=np.ndarray([
        #         [0, 0], [0, 1]
        #     ]),
        #     color=(1, 1, 1),
        #     parent=view.scene
        # )

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

    # ----- Closing events -----

    def closeEvent(self, event):

        # stop all threads

        try:
            self.imageProcess.quit()
            self.imageProcess.wait()
        except AttributeError:
            ...

        try:
            self.vectorScopeProcess.quit()
            self.vectorScopeProcess.wait()
        except AttributeError:
            ...

        try:
            self.imageProcess.quit()
            self.imageProcess.wait()
        except AttributeError:
            ...

        try:
            self.grabber.quit()
            self.grabber.wait()
        except AttributeError:
            ...

        try:
            vispy.app.quit()
        except Exception as e:
            logging.error(e)

    # ------ Image processing ------

    def drawImg(self, image: np.ndarray):
        if IMAGE_DRAW_METHOD == "QtImageViewer":
            qImg = QtGui.QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.shape[1] * image.shape[2],
                QtGui.QImage.Format_RGB888
            )
            self.imageViewer.setImage(qImg.rgbSwapped())
        elif IMAGE_DRAW_METHOD == "Vispy":
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # swapped image ?
            self.imageViewerPhoto.set_data(image)
            self.imageViewerPhoto.update()
            self.imageViewerCanvas.update()

    # def drawQimg(self, qImg: QtGui.QImage):
    #     self.imageViewer.setImage(qImg.rgbSwapped())

    def clbkProcessImage(self, binImage):
        self.imageProcess.setImg(np.copy(binImage))
        self.imageProcess.process()

    def updateHistogram(self, binImage):
        self.histogramProcess.setImg(binImage)
        self.histogramProcess.process()

    def updateHistogramWidget(self, dataList):
        x, y = dataList
        yR, yG1, yG2, yB = y
        # self.histographPlot.plot(x, y, stepMode=True, clear=True)
        self.histographPlot.plot(x, yR, stepMode=True, pen=(255, 0, 0), clear=True)
        self.histographPlot.plot(x, yG1, stepMode=True, pen=(0, 255, 0))
        self.histographPlot.plot(x, yG2, stepMode=True, pen=(0, 255, 0))
        self.histographPlot.plot(x, yB, stepMode=True, pen=(0, 0, 255))

    def updateVectorScope(self, binImage):
        if not self.vectorScopeProcess.isRunning():
            self.vectorScopeProcess.setImg(binImage)
            self.vectorScopeProcess.process()

    def updateVectorScopeWidget(self, vectorScopeData):
        pos, colors = vectorScopeData
        self.vectorScopePlot.set_data(
            pos * 20, face_color=colors, symbol='o', size=2,
            edge_width=0, edge_color=None, scaling=False
        )

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
        self.mode.setCurrentIndex(
            self.mode.findData(self.ia.remote_device.node_map.VideoMode.value)
        )

    def clbkUpdateUiLimits(self):
        self.horizontalResolution.setMaximum(self.ia.remote_device.node_map.WidthMax.value)
        self.verticalResolution.setMaximum(self.ia.remote_device.node_map.HeightMax.value)

    def clbkRunCamera(self):
        if self.ia.is_acquiring():
            self.ia.stop_acquisition()
            self.runButton.setText("Run")
            self.recordButton.setEnabled(False)
            self.imageProcess.quit()
            self.imageProcess.wait()
            self.grabber.quit()
            self.grabber.wait()
        else:
            self.ia.start_acquisition()
            self.initAcquisitionThread()
            self.grabber.process()
            self.initImageProcessThread()
            self.imageProcess.setLUT(self.LUT)
            self.runButton.setText("Stop")
            self.recordButton.setEnabled(True)
            self.updateRGBGain()

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

    def updateRGBGain(self):
        redGain = self.redGain.value() / 100
        self.redGammaLabel.setText("R : " + str(redGain))
        greenGain = self.greenGamma.value() / 100
        self.greenGammaLabel.setText("G : " + str(greenGain))
        blueGain = self.blueGain.value() / 100
        self.blueGammaLabel.setText("B : " + str(blueGain))
        self.imageProcess.setGamma(redGain, greenGain, blueGain)


def SWCamera():
    app = QtWidgets.QApplication(sys.argv)
    _gui = SWCameraGui()  # noqa F841
    if VECTORSCOPE:
        vispy.app.run()

    sys.exit(app.exec_())


if __name__ == '__main__':
    SWCamera()
