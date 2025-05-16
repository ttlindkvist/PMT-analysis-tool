import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class CanvasWidget(QWidget):
    def __init__(self, *args, **kwargs):
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes.tick_params(which='major', axis='both', length=10, width=1.2)
        self.sc.axes.tick_params(which='minor', axis='both', length=6, width=1.2)
        toolbar = NavigationToolbar(self.sc, self)

        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)
        super(QWidget, self).__init__(*args, **kwargs)
        self.setLayout(layout)
    
    def clear(self):
        self.sc.axes.cla()
    def plot(self, *args, **kwargs):
        self.sc.axes.plot(*args, **kwargs)