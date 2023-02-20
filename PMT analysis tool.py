import sys, os, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QPushButton, QCheckBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

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
        
        # data_dir_path = "E:\\LUNA2\\PMTTestData"
        # data_dir_path = 'O:\\Nat_IFA-fullc\\people\\akThomas\\Sum spectra\\'
        data_dir_path = 'O:\\Nat_IFA-fullc\\experimental\\expdata\\Fluorescence_LUNA2\\PMTdata_summed\\'
        self.datadirtree = DataDirTree(data_dir_path)
        right_layout.addWidget(self.datadirtree.tree)
        
        sum_buttons = QHBoxLayout()
        sum_selected = QPushButton("Sum selected")
        sum_selected.clicked.connect(self.sum_button_clicked)
        button2 = QPushButton("Reset sums")
        sum_buttons.addWidget(sum_selected)
        sum_buttons.addWidget(button2)
        right_layout.addLayout(sum_buttons)

        self.sum_tree = QTreeWidget()
        self.sum_tree.setColumnCount(2)
        self.sum_tree.setHeaderLabels(["Sum selection", "Runs"])
        right_layout.addWidget(self.sum_tree)
        
        display_check_boxes_layout = QHBoxLayout()
        self.use_scale_cb = QCheckBox("Rescale")
        self.normalize_cb = QCheckBox("Normalize")
        self.reciprocal_cm = QCheckBox("cm^-1")
        self.reciprocal_cm_zero = QLineEdit("500")
        self.reciprocal_cm_zero.setValidator(QDoubleValidator())
        display_check_boxes_layout.addWidget(self.use_scale_cb)
        display_check_boxes_layout.addWidget(self.normalize_cb)
        display_check_boxes_layout.addWidget(self.reciprocal_cm_zero)
        display_check_boxes_layout.addWidget(self.reciprocal_cm)
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
        it = QTreeWidgetItemIterator(self.sum_tree)
        n_sums = 0
        while it.value():
            n_sums += 1
            it += 1
        if len(selected_runs) == 0 and n_sums == 0:
            return
        
        self.datahandler.load_runs(selected_runs)
        self.sc.axes.cla()  # Clear the canvas.
        
        wl_range = [np.inf, -np.inf]
        signal_range = [np.inf, -np.inf]
        
        #Individual runs
        for selected_run, scale in zip(selected_runs, scales):
            wls = self.datahandler.absorption_spectra[selected_run]['wavelengths']
            abs = self.datahandler.absorption_spectra[selected_run]['absorption']
            
            if self.normalize_cb.isChecked():
                abs = normalize(abs)
            if self.use_scale_cb.isChecked():
                abs = abs*scale
                
            #Update plot ranges
            if np.min(wls) < wl_range[0]: wl_range[0] = np.min(wls)
            if np.max(wls) > wl_range[1]: wl_range[1] = np.max(wls)
            if np.min(abs) < signal_range[0]: signal_range[0] = np.min(abs)
            if np.max(abs) > signal_range[1]: signal_range[1] = np.max(abs)
            
            if not self.reciprocal_cm.isChecked():
                self.sc.axes.plot(wls, abs, label=os.path.basename(selected_run))
            else:
                zero_rec_cm_nm = float(self.reciprocal_cm_zero.text())
                self.sc.axes.plot((1/wls - 1/zero_rec_cm_nm)*1e7, abs, label=os.path.basename(selected_run))
        
        #Plot sum spectra
        it = QTreeWidgetItemIterator(self.sum_tree, QTreeWidgetItemIterator.IteratorFlag.Checked)
        while it.value():
            sum_name = it.value().text(0)
            complete_runs_string = it.value().data(1, Qt.ItemDataRole.UserRole)
            wls = self.datahandler.absorption_spectra[complete_runs_string]['wavelengths']
            abs = self.datahandler.absorption_spectra[complete_runs_string]['absorption']
            abs_std = self.datahandler.absorption_spectra[complete_runs_string]['absorption_std']
            
            if self.normalize_cb.isChecked():
                abs = normalize(abs)
                
            #Update plot ranges
            if np.min(wls) < wl_range[0]: wl_range[0] = np.min(wls)
            if np.max(wls) > wl_range[1]: wl_range[1] = np.max(wls)
            if np.min(abs) < signal_range[0]: signal_range[0] = np.min(abs)
            if np.max(abs) > signal_range[1]: signal_range[1] = np.max(abs)
            
            if not self.reciprocal_cm.isChecked():
                self.sc.axes.plot(wls, abs, label=sum_name)
            else:
                zero_rec_cm_nm = float(self.reciprocal_cm_zero.text())
                self.sc.axes.plot((1/wls - 1/zero_rec_cm_nm)*1e7, abs, label=sum_name)
            
            it += 1
        
        if not self.reciprocal_cm.isChecked():
            self.sc.axes.hlines(0, 0, 1000, ls='--', colors='k')
            self.sc.axes.set_xlim(*wl_range)
            self.sc.axes.set_xlabel('Wavelength (nm)')
        else:
            zero_rec_cm_nm = float(self.reciprocal_cm_zero.text())
            self.sc.axes.hlines(0, -100000, 100000, ls='--', colors='k')
            self.sc.axes.set_xlim((1/wl_range[1]-1/zero_rec_cm_nm)*1e7, (1/wl_range[0]-1/zero_rec_cm_nm)*1e7)
            self.sc.axes.set_xlabel('Wavenumber (cm$^{-1}$)')
        
        self.sc.axes.set_ylim(min(signal_range[0], 0), signal_range[1]*1.05)
        self.sc.axes.legend(frameon=False)
        self.sc.axes.set_ylabel('Normalized signal' if self.normalize_cb.isChecked() else 'Signal')
        self.sc.figure.tight_layout() 
        self.sc.draw()

    def sum_button_clicked(self):
        selected_runs, scales = self.datadirtree.getSelectedRuns()
        if len(selected_runs) == 0:
            return
        #Calculate sum spectrum
        self.datahandler.sum_runs(selected_runs, run_scalings=scales)
        
        runs_string = ''
        complete_runs_string = ''
        for selected_run in selected_runs:
            runs_string += os.path.basename(selected_run)+' '
            complete_runs_string += selected_run
        
        # #Get number of sums
        it = QTreeWidgetItemIterator(self.sum_tree)
        n_sums = 0
        while it.value():
            n_sums += 1
            it += 1
        
        #Add sum spectrum to tree
        sum_item = QTreeWidgetItem(self.sum_tree)
        sum_item.setText(0, 'Sum {}'.format(n_sums+1))
        sum_item.setText(1, runs_string)
        sum_item.setData(1, Qt.ItemDataRole.UserRole, complete_runs_string)
        sum_item.setCheckState(0, Qt.CheckState.Unchecked)
        sum_item.setFlags(sum_item.flags() | Qt.ItemFlag.ItemIsEditable)
    
app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
