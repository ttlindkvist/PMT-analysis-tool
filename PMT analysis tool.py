import sys, os, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QPushButton, QCheckBox
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from DataDirTree import *
from DataHandler import *

def normalize(data):
    return data/np.max(data)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)

        

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        # create a widget to hold the plot and toolbar on the left
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(toolbar)
        left_layout.addWidget(self.sc)
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        self.datadirtree = DataDirTree()
        right_layout.addWidget(self.datadirtree.tree)
        
        button_layout = QHBoxLayout()
        button1 = QPushButton("Sum selected")
        button2 = QPushButton("Reset sums")
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        right_layout.addLayout(button_layout)

        tree_widget_2 = QTreeWidget()
        tree_widget_2.setHeaderLabels(["Column 1", "Column 2"])
        right_layout.addWidget(tree_widget_2)
        
        display_check_boxes_layout = QHBoxLayout()
        self.use_scale_cb = QCheckBox("Rescale")
        self.normalize_cb = QCheckBox("Normalize")
        display_check_boxes_layout.addWidget(self.use_scale_cb)
        display_check_boxes_layout.addWidget(self.normalize_cb)
        right_layout.addLayout(display_check_boxes_layout)

        display_button = QPushButton("Display selected")
        display_button.clicked.connect(self.display_button_clicked)
        right_layout.addWidget(display_button)

        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(600)
        main_layout.addWidget(left_widget, 7)
        main_layout.addWidget(right_widget, 3)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)        
        self.show()
        
        # Create data handler object
        self.datahandler = DataHandler()

    def display_button_clicked(self):
        selected_runs, scales = self.datadirtree.getSelectedRuns()
        if len(selected_runs) == 0:
            return
        self.datahandler.load_runs(selected_runs)
        self.sc.axes.cla()  # Clear the canvas.
        
        wl_range = [np.inf, -np.inf]
        signal_range = [np.inf, -np.inf]
        for selected_run, scale in zip(selected_runs, scales):
            wls = self.datahandler.cached_runs[selected_run][0]
            if np.min(wls) < wl_range[0]: wl_range[0] = np.min(wls)
            if np.max(wls) > wl_range[1]: wl_range[1] = np.max(wls)
            
            abs = self.datahandler.absorptions[selected_run]
            if self.normalize_cb.isChecked():
                abs = normalize(abs)
            if self.use_scale_cb.isChecked():
                abs = abs*scale
                
            if np.min(abs) < signal_range[0]: signal_range[0] = np.min(abs)
            if np.max(abs) > signal_range[1]: signal_range[1] = np.max(abs)
            self.sc.axes.plot(wls, abs, label=os.path.basename(selected_run))
            
        self.sc.axes.hlines(0, 0, 1000, ls='--')
        self.sc.axes.set_xlim(*wl_range)
        self.sc.axes.set_ylim(min(signal_range[0], 0), signal_range[1]*1.05)
        self.sc.axes.legend(frameon=False)
        self.sc.axes.set_xlabel('Wavelength (nm)')
        self.sc.axes.set_ylabel('Normalized signal' if self.normalize_cb.isChecked() else 'Signal')
        self.sc.figure.tight_layout() 
        self.sc.draw()
        
        

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
