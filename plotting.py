import cv2
from cv2 import data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_agg
import pyqtgraph as pg

plt.style.use('dark_background')


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


def vectorscope(cbData, crData, colors):
    fig = plt.figure()
    canvas = matplotlib.backends.backend_agg.FigureCanvas(fig)
    ax = fig.subplots()
    ax.scatter(cbData, crData, marker=',', s=1, c=colors)
    ax.set_aspect(aspect=1.0)
    MAX = 1
    ax.set_xlim([-0.05, MAX + 0.05])
    ax.set_ylim([-0.05, MAX + 0.05])

    # axis
    ax.hlines(0.5, 0, 1, linewidth=1)
    ax.vlines(0.5, 0, 1, linewidth=1)

    # reference colors
    calibImgRgb = np.zeros((3, 3, 3))
    calibImgRgb[0, 0] = [1.0, 0.0, 0.0]
    calibImgRgb[0, 1] = [1.0, 1.0, 0.0]
    calibImgRgb[0, 2] = [0.0, 1.0, 0.0]
    calibImgRgb[1, 0] = [0.0, 1.0, 1.0]
    calibImgRgb[1, 1] = [0.0, 0.0, 1.0]
    calibImgRgb[1, 2] = [1.0, 0.0, 1.0]
    calibImgRgb[2, 0] = [1.0, 0.0, 0.0]

    calibImgYCrCb = cv2.cvtColor(calibImgRgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    ax.plot(calibImgYCrCb[:, :, 2].flatten()[:7], calibImgYCrCb[:, :, 1].flatten()[:7], '-', linewidth=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Cb")
    ax.set_ylabel("Cr")

    ax.text(calibImgYCrCb[0, 0, 2], calibImgYCrCb[0, 0, 1], "R", ha="right", va="bottom")
    ax.text(calibImgYCrCb[0, 1, 2], calibImgYCrCb[0, 1, 1], "Y", ha="right", va="center")
    ax.text(calibImgYCrCb[0, 2, 2], calibImgYCrCb[0, 2, 1], "G", ha="right", va="top")
    ax.text(calibImgYCrCb[1, 0, 2], calibImgYCrCb[1, 0, 1], "C", ha="center", va="top")
    ax.text(calibImgYCrCb[1, 1, 2], calibImgYCrCb[1, 1, 1], "B", ha="left", va="top")
    ax.text(calibImgYCrCb[1, 2, 2], calibImgYCrCb[1, 2, 1], "M", ha="left", va="bottom")

    ax.plot([0.0, 1.0], [1.0, 0.0], ':', linewidth=1, color="#ffffffaa")

    canvas.draw()

    plotImage = np.array(canvas.renderer.buffer_rgba())

    plt.show()

    return plotImage


def pyqtVectorScope(cbData, crData, color, scatterPlotItem):
    scatterPlotItem.clear()

    spots = [
        {
            "pos": [cbData[i], crData[i]],
            "data": 1,
            "size": 4,
            "brush": tuple(colors[i] * 255)
        }
        for i in range(len(cbData))]

    scatterPlotItem.addPoints(spots, pxMode=True)


def removeDuplicates(l):
    b = []
    for i in range(0, len(l)):
        if l[i] not in l[i+1:]:
            b.append(l[i])
    return b


if __name__ == "__main__":

    from PySide2 import QtGui
    app = pg.mkQApp("Scatter Plot Item Example")
    mw = QtGui.QMainWindow()
    view = pg.GraphicsLayoutWidget()  # GraphicsView with GraphicsLayout inserted by default
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle('pyqtgraph example: ScatterPlot')
    w1 = view.addPlot()
    w1.setXRange(0, 1)
    w1.setYRange(0, 1)
    s1 = pg.ScatterPlotItem(pen=pg.mkPen(None))
    w1.addItem(s1)

    img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgbImg = cv2.resize(rgbImg, (320, 240), cv2.INTER_AREA)
    cbData, crData, colors = extractCbCrData(rgbImg)
    pyqtVectorScope(cbData, crData, colors, s1)

    pg.mkQApp().exec_()
