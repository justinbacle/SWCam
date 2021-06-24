import cv2
import numpy as np
import pyqtgraph as pg
from PySide2 import QtGui
import vispy
import vispy.scene
import sys
import os

sys.path.append(os.getcwd())
import plotting
import vectorscope


def pyqtVectorScope_test(cbData, crData, colors):

    # app = pg.mkQApp("Scatter Plot Ite>Zm Example")
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

    plotting.pyqtVectorScope(cbData, crData, colors, s1)

    pg.mkQApp().exec_()


def vispyVectorScope_test(cbData, crData, colors):
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 0
    view.camera.set_range(
        x=(-5, 5),
        y=(-5, 5),
        margin=0.0
    )
    view.camera.interactive = False

    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.antialias = 0

    pos = np.transpose(np.vstack(
        (
            cbData-0.5,
            crData-0.5,
            np.zeros_like(cbData)
        )
    ))

    p1.set_data(pos * 20, face_color=colors, symbol='o', size=2,
                edge_width=0, edge_color=None, scaling=False)


if __name__ == "__main__":

    img = cv2.imread("C:/Users/justi/OneDrive/Images/Xbox Screenshots/03-02-2018_22-33-56.png")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rgbImg = cv2.resize(rgbImg, (960, 540), cv2.INTER_AREA)
    cbData, crData, colors = vectorscope.extractCbCrData(rgbImg)

    # pyqtVectorScope_test(cbData, crData, colors)
    vispyVectorScope_test(cbData, crData, colors)

    if sys.flags.interactive != 1:
        vispy.app.run()
