# -*- coding: utf-8 -*-
"""
Example demonstrating a variety of scatter plot features.
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


app = pg.mkQApp("Scatter Plot Item Example")
mw = QtGui.QMainWindow()
view = pg.GraphicsLayoutWidget()  # GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')


w1 = view.addPlot()

n = int(1e4)
s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
pos = np.random.normal(size=(2, n), scale=1e-5)
spots = [
    {
        'pos': pos[:, i],
        'data': 1,
        'size': 4,
        'brush': (100, 120, 150, 150),
    }
    for i in range(n)]
s1.addPoints(spots)
w1.addItem(s1)

if __name__ == '__main__':
    pg.mkQApp().exec_()
