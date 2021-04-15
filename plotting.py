import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_agg
import pyqtgraph as pg
from PySide2 import QtGui
import vispy
import vispy.scene
import sys

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


def pyqtVectorScope(cbData, crData, colors, scatterPlotItem):
    spots = [
        {
            "pos": [cbData[i], crData[i]],
            "data": 1,
            "size": 1,
            "brush": tuple(colors[i] * 255)
        }
        for i in range(len(cbData))]
    scatterPlotItem.setData(
        spots, pxMode=True, compositionMode=QtGui.QPainter.CompositionMode_SoftLight, _callSync='off')


def pyqtVectorScope_test(cbData, crData, colors):

    app = pg.mkQApp("Scatter Plot Item Example")
    mw = QtGui.QMainWindow()
    view = pg.GraphicsLayoutWidget()  # GraphicsView with GraphicsLayout inserted by default
    mw.setCentralWidget(view)
    mw.show()
    mw.setWindowTitle('Vectorscope')
    w1 = view.addPlot()
    w1.setXRange(0, 1)
    w1.setYRange(0, 1)
    s1 = pg.ScatterPlotItem(pen=pg.mkPen(None))
    w1.addItem(s1)

    pyqtVectorScope(cbData, crData, colors, s1)

    pg.mkQApp().exec_()


def vispyVectorScope_test(cbData, crData, colors):
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 0
    view.camera.set_range(
        x=(0, 1),
        y=(0, 1),
        margin=0.0
    )
    view.camera.interactive = False

    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.antialias = 0

    pos = np.transpose(np.vstack(
        (
            cbData,
            crData,
            np.zeros_like(cbData)
        )
    ))

    p1.set_data(pos, face_color=colors, symbol='o', size=2,
                edge_width=0, edge_color=None, scaling=False)


if __name__ == "__main__":

    img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgbImg = cv2.resize(rgbImg, (960, 540), cv2.INTER_AREA)
    cbData, crData, colors = extractCbCrData(rgbImg)

    # pyqtVectorScope_test(cbData, crData, colors)
    vispyVectorScope_test(cbData, crData, colors)

    if sys.flags.interactive != 1:
        vispy.app.run()
