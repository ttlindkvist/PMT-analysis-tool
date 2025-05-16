import sys, os, glob
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('qtagg')

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QTabWidget, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout,\
                            QWidget, QPushButton, QCheckBox, QLineEdit, QTreeWidgetItemIterator, QLabel, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PMTDirTree import *
from SpectrometerDirTree import *
from DataHandler import *
from SpecDataHandler import *
from DataDirTree import EditableDelegate
from CanvasWidget import CanvasWidget

def normalize(data):
    return data/np.nanmax(data)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('PMT analysis tool')

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes = self.sc.fig.add_subplot(4,1,(1,3))
        self.sc.axes.tick_params(which='major', axis='both', length=10, width=1.2)
        self.sc.axes.tick_params(which='minor', axis='both', length=6, width=1.2)
        self.sc.laser_ax = self.sc.fig.add_subplot(4, 1, (4))
        self.sc.laser_ax.sharex(self.sc.axes)
        toolbar = NavigationToolbar(self.sc, self)

        #Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(toolbar)
        left_layout.addWidget(self.sc)
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        PMT_data_dir_path = 'O:\\Nat_IFA-fullc\\experimental\\expdata\\Fluorescence_LUNA2\\PMTdata_summed\\'
        spec_data_dir_path = 'O:\\Nat_IFA-fullc\\experimental\\expdata\\Fluorescence_LUNA2\\DATA\\'

        # Create data handler object
        self.datahandler = DataHandler()
        self.spec_datahandler = SpecDataHandler()

        self.datadirtree = PMTDirTree(PMT_data_dir_path)
        self.spec_data_dir_tree = SpectrometerDirTree(spec_data_dir_path, self.spec_datahandler)

        self.dir_tabs = QTabWidget()
        self.dir_tabs.addTab(self.datadirtree.tree, 'PMT data')
        self.dir_tabs.addTab(self.spec_data_dir_tree.tree, 'Spectrometer data')
        right_layout.addWidget(self.dir_tabs)
        
        k_calib_layout = QHBoxLayout()
        self.k_calib_cb = QCheckBox('Use power calibration')
        self.k_calib_cb.setChecked(True)
        self.k_calib_filename = QLineEdit('O:\\Nat_IFA-fullc\\experimental\\expdata\\Fluorescence_LUNA2\\Photo diode calibration\\AbsoluteCalibFiles\\PDabsoluteCalib_240617.txt')
        k_calib_file_browse = QPushButton('Browse')
        k_calib_file_browse.clicked.connect(self.k_calib_open_file_dialog)

        k_calib_layout.addWidget(self.k_calib_cb)
        k_calib_layout.addWidget(QLabel('File:'))
        k_calib_layout.addWidget(self.k_calib_filename)
        k_calib_layout.addWidget(k_calib_file_browse)

        display_buttons1 = QHBoxLayout()
        display_button = QPushButton("Display selected")
        display_button.clicked.connect(self.display_button_clicked)
        clear_selection_button = QPushButton("Clear selection")
        clear_selection_button.clicked.connect(self.clear_selection_button_clicked)
        autoscale_button = QPushButton("Autoscale selected")
        autoscale_button.clicked.connect(self.autoscale_button_clicked)
        reload_files_button = QPushButton('Reload selected')
        reload_files_button.clicked.connect(self.reload_selected_runs)
        display_buttons1.addWidget(display_button)
        display_buttons1.addWidget(clear_selection_button)
        display_buttons1.addWidget(autoscale_button)
        # display_buttons1.addWidget(reload_files_button)
        right_layout.addLayout(display_buttons1)

        right_layout.addLayout(k_calib_layout)

        binning_layout = QHBoxLayout()
        self.use_auto_scale_cb = QCheckBox('Use autoscale')
        self.sum_bin_width = QLineEdit('1')
        self.sum_bin_width.setValidator(QDoubleValidator())
        self.sum_bin_width.setFixedWidth(60)
        self.sum_bin_cb = QCheckBox('(nm) Use sum binning')
        self.sum_bin_cb.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        sum_selected_button = QPushButton("Sum selected")
        sum_selected_button.clicked.connect(self.sum_button_clicked)
        binning_layout.addWidget(self.use_auto_scale_cb)
        binning_layout.addWidget(self.sum_bin_width)
        binning_layout.addWidget(self.sum_bin_cb)
        binning_layout.addWidget(sum_selected_button)
        right_layout.addLayout(binning_layout)
        
        sum_buttons2 = QHBoxLayout()
        reset_sum_button = QPushButton("Reset sums")
        reset_sum_button.clicked.connect(self.reset_sum_button_clicked)
        refresh_trees_button = QPushButton("Refresh")
        refresh_trees_button.clicked.connect(self.refresh_trees_button_clicked)
        self.force_sum_cb = QCheckBox('Force sum')
        sum_buttons2.addWidget(reset_sum_button)
        sum_buttons2.addWidget(refresh_trees_button)
        sum_buttons2.addWidget(self.force_sum_cb)
        right_layout.addLayout(sum_buttons2)

        self.sum_tree = QTreeWidget()
        self.sum_tree_headers = {'Sum selection': 0, 'Scale': 1, 'Molecule': 2, 'Runs': 3}
        self.sum_tree.setColumnCount(len(self.sum_tree_headers.keys()))
        self.sum_tree.setHeaderLabels(self.sum_tree_headers.keys())
        self.sum_tree.setItemDelegateForColumn(self.sum_tree_headers['Scale'], EditableDelegate(self.sum_tree))
        self.sum_tree.itemDoubleClicked.connect(self.sum_item_double_clicked)
        right_layout.addWidget(self.sum_tree)
        
        display_check_boxes_layout = QHBoxLayout()
        self.use_scale_cb = QCheckBox("Rescale")
        self.normalize_cb = QCheckBox("Normalize")
        self.reciprocal_cm = QCheckBox("cm^-1")
        self.reciprocal_cm_zero = QLineEdit("500")
        self.reciprocal_cm_zero.setValidator(QDoubleValidator())
        display_check_boxes_layout.addWidget(self.use_scale_cb)
        display_check_boxes_layout.addWidget(self.normalize_cb)
        display_check_boxes_layout.addWidget(self.reciprocal_cm)
        display_check_boxes_layout.addWidget(self.reciprocal_cm_zero)
        right_layout.addLayout(display_check_boxes_layout)

        display_check_boxes_layout2 = QHBoxLayout()
        self.only_sum_cb = QCheckBox('Only sum spectra')
        self.uncertainty_cb = QCheckBox('Stat uncertainty')
        display_check_boxes_layout2.addWidget(self.only_sum_cb)
        display_check_boxes_layout2.addWidget(self.uncertainty_cb)
        right_layout.addLayout(display_check_boxes_layout2)

        display_check_boxes_layout3 = QHBoxLayout()
        self.display_powercalib_cb = QCheckBox('Use power calib')
        self.display_laser_calib_cb = QCheckBox('Use laser wl calib')
        self.laser_combo_box = QComboBox()
        self.laser_combo_box.addItem('ULLA')
        self.laser_combo_box.addItem('VICTOR')
        display_check_boxes_layout3.addWidget(self.display_powercalib_cb)
        display_check_boxes_layout3.addWidget(self.display_laser_calib_cb)
        display_check_boxes_layout3.addWidget(self.laser_combo_box)
        right_layout.addLayout(display_check_boxes_layout3)

        display_button = QPushButton("Display selected")
        display_button.clicked.connect(self.display_button_clicked)
        right_layout.addWidget(display_button)

        export_button = QPushButton("Export selected")
        export_button.clicked.connect(self.export_button_clicked)
        right_layout.addWidget(export_button)

        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(600)
        main_layout.addWidget(left_widget, 7)
        main_layout.addWidget(right_widget, 3)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)        
        self.resizeEvent = self.on_resize
        self.show()
        

    def display_button_clicked(self):
        selected_runs, scales = self.datadirtree.getSelectedRuns()
        disp_selected_runs, disp_scalings = self.spec_data_dir_tree.getSelectedRuns()

        it = QTreeWidgetItemIterator(self.sum_tree, QTreeWidgetItemIterator.IteratorFlag.Checked)
        n_sums_checked = 0
        while it.value():
            n_sums_checked += 1
            it += 1
        if len(selected_runs) == 0 and n_sums_checked == 0 \
            and len(disp_selected_runs) == 0:
            return
        if self.only_sum_cb.isChecked() and n_sums_checked == 0:
            return
        if not self.only_sum_cb.isChecked():
            self.datahandler.load_runs(selected_runs, self.k_calib_cb.isChecked(), self.k_calib_filename.text())
        
        self.sc.axes.cla()  # Clear the canvas.
        self.sc.laser_ax.cla()  # Clear the canvas.
        
        wl_range = [np.inf, -np.inf]
        signal_range = [np.inf, -np.inf]
        
        #Individual runs - PMT
        if not self.only_sum_cb.isChecked():
            for selected_run, scale in zip(selected_runs, scales):
                wls = self.datahandler.absorption_spectra[selected_run]['wavelengths']
                abs = self.datahandler.absorption_spectra[selected_run]['absorption']
                pd_power = self.datahandler.absorption_spectra[selected_run]['PD power']
                if not self.display_powercalib_cb.isChecked():
                    abs = self.datahandler.absorption_spectra[selected_run]['absorption (not power calibrated)']
                if self.display_laser_calib_cb.isChecked(): wls = calibrate_laser(wls, self.laser_combo_box.currentText())
                if self.use_auto_scale_cb.isChecked():
                    auto_scaling_factor = self.datahandler.absorption_spectra[selected_run].get("autoscaling factor", 1)
                    abs = abs * auto_scaling_factor
                if self.normalize_cb.isChecked():
                    abs = normalize(abs)
                if self.use_scale_cb.isChecked():
                    abs = abs*scale
                    
                #Update plot ranges
                if np.nanmin(wls) < wl_range[0]: wl_range[0] = np.nanmin(wls)
                if np.nanmax(wls) > wl_range[1]: wl_range[1] = np.nanmax(wls)
                if np.nanmin(abs) < signal_range[0]: signal_range[0] = np.nanmin(abs)
                if np.nanmax(abs) > signal_range[1]: signal_range[1] = np.nanmax(abs)
                
                xs = wls[:]
                if self.reciprocal_cm.isChecked():
                    zero_rec_cm_nm = float(self.reciprocal_cm_zero.text().replace(',','.'))
                    xs = (1/wls - 1/zero_rec_cm_nm)*1e7
                self.sc.axes.plot(xs, abs, label=os.path.basename(selected_run))
                self.sc.laser_ax.plot(xs, pd_power, label=os.path.basename(selected_run))
        
        ##### Dispersed fluorescence #####
        # individual runs
        self.spec_datahandler.load_runs(disp_selected_runs)
        if not self.only_sum_cb.isChecked():
            for selected_run, scale in zip(disp_selected_runs, disp_scalings):
                wls = self.spec_datahandler.dispersed_fluorescence[selected_run]['wavelengths']
                disp_fluor = self.spec_datahandler.dispersed_fluorescence[selected_run]['fluorescence']
                if self.use_auto_scale_cb.isChecked():
                    auto_scaling_factor = self.spec_datahandler.dispersed_fluorescence[selected_run].get("autoscaling factor", 1)
                    disp_fluor = disp_fluor * auto_scaling_factor
                if self.normalize_cb.isChecked():
                    disp_fluor = normalize(disp_fluor)
                if self.use_scale_cb.isChecked():
                    disp_fluor = disp_fluor*scale
                    
                #Update plot ranges
                if np.nanmin(wls) < wl_range[0]: wl_range[0] = np.nanmin(wls)
                if np.nanmax(wls) > wl_range[1]: wl_range[1] = np.nanmax(wls)
                if np.nanmin(disp_fluor) < signal_range[0]: signal_range[0] = np.nanmin(disp_fluor)
                if np.nanmax(disp_fluor) > signal_range[1]: signal_range[1] = np.nanmax(disp_fluor)
                
                if not self.reciprocal_cm.isChecked():
                    self.sc.axes.plot(wls, disp_fluor, label=os.path.basename(selected_run))
                else:
                    zero_rec_cm_nm = float(self.reciprocal_cm_zero.text().replace(',','.'))
                    self.sc.axes.plot((1/wls - 1/zero_rec_cm_nm)*1e7, disp_fluor, label=os.path.basename(selected_run))
        
        #Plot sum spectra - both PMT and iCCD
        it = QTreeWidgetItemIterator(self.sum_tree, 
                                     QTreeWidgetItemIterator.IteratorFlag.Checked | QTreeWidgetItemIterator.IteratorFlag.NoChildren)
        while it.value():
            sum_name = it.value().text(self.sum_tree_headers['Sum selection'])
            scale = it.value().data(self.sum_tree_headers['Scale'], Qt.ItemDataRole.EditRole)
            sum_data_key = it.value().data(self.sum_tree_headers['Runs'], Qt.ItemDataRole.UserRole)
            it += 1
            
            wls = np.nan
            signal = np.nan
            signal_std = np.nan
            molecule_str = ''
            
            if sum_data_key.split()[0] == 'PMT':
                abs_spectrum = self.datahandler.absorption_spectra[sum_data_key]
                wls = abs_spectrum['wavelengths']
                if self.display_laser_calib_cb.isChecked(): wls = calibrate_laser(wls, self.laser_combo_box.currentText())
                signal = abs_spectrum['absorption']
                signal_std = abs_spectrum['absorption_std']
                molecule_str = abs_spectrum['molecule']
            if sum_data_key.split()[0] == 'iCCD':
                disp_spectrum = self.spec_datahandler.dispersed_fluorescence[sum_data_key] 
                wls = disp_spectrum['wavelengths']
                signal = disp_spectrum['fluorescence']
                signal_std = disp_spectrum['fluorescence_std']
                molecule_str = disp_spectrum['molecule']
                
            # Normalize signal
            if self.normalize_cb.isChecked():
                signal_std = signal_std/np.nanmax(signal)
                signal = signal/np.nanmax(signal)

            # Manual rescaling of signal
            if self.use_scale_cb.isChecked():
                signal = signal*scale
                signal_std = signal_std*scale
                
            #Update plot ranges
            if np.nanmin(wls) < wl_range[0]: wl_range[0] = np.nanmin(wls)
            if np.nanmax(wls) > wl_range[1]: wl_range[1] = np.nanmax(wls)
            if np.nanmin(signal) < signal_range[0]: signal_range[0] = np.nanmin(signal)
            if np.nanmax(signal) > signal_range[1]: signal_range[1] = np.nanmax(signal)
            

            xs = wls
            if self.reciprocal_cm.isChecked():
                zero_rec_cm_nm = float(self.reciprocal_cm_zero.text().replace(',','.'))
                xs = (1/wls - 1/zero_rec_cm_nm)*1e7
                # self.sc.axes.plot((1/wls - 1/zero_rec_cm_nm)*1e7, signal, label=sum_name+' '+molecule_str)
            
            p = self.sc.axes.plot(xs, signal, label=sum_name+' '+molecule_str)
            if self.uncertainty_cb.isChecked():
                self.sc.axes.fill_between(xs, signal-signal_std, signal+signal_std, color=p[0].get_color(), alpha=0.75)
            
        #Update x-axis limits and plot zero-line
        # In wavelength- or energy-space 
        if not self.reciprocal_cm.isChecked():
            self.sc.axes.hlines(0, *wl_range, ls='--', colors='k')
            self.sc.axes.set_xlim(*wl_range)
            self.sc.axes.set_xlabel('Wavelength (nm)')
        else:
            zero_rec_cm_nm = float(self.reciprocal_cm_zero.text().replace(',','.'))
            rec_x_range = (1/wl_range[1]-1/zero_rec_cm_nm)*1e7, (1/wl_range[0]-1/zero_rec_cm_nm)*1e7
            self.sc.axes.hlines(0, -100000, 100000, ls='--', colors='k')
            self.sc.axes.set_xlim(*rec_x_range)
            self.sc.axes.set_xlabel('Wavenumber (cm$^{-1}$)')
        self.sc.axes.set_ylim(min(signal_range[0], 0), signal_range[1]*1.05)
        self.sc.laser_ax.set_ylim(0, self.sc.laser_ax.get_ylim()[1])
        self.sc.laser_ax.set_ylabel('PD power (arb. u.)')
        
        self.update_ticks()

        self.sc.axes.legend(frameon=False)
        self.sc.axes.set_ylabel('Normalized signal' if self.normalize_cb.isChecked() else 'Signal')
        self.sc.figure.tight_layout()
        self.sc.draw()

    def sum_button_clicked(self):
        autoscalings = []
        selected_runs = []
        scalings = []
        molecules = []
        
        tab_index = self.dir_tabs.currentIndex()
        TAB_ENUM = {0: 'PMT', 1 : 'iCCD'}
        if TAB_ENUM[tab_index] == 'PMT':
            selected_runs, scalings = self.datadirtree.getSelectedRuns()
            self.datahandler.load_runs(selected_runs, self.k_calib_cb.isChecked(), self.k_calib_filename.text())
            for selected_run in selected_runs:
                
                autoscalings.append(self.datahandler.absorption_spectra[selected_run].get("autoscaling factor", 1) if self.use_auto_scale_cb.isChecked() 
                                    else 1)
                molecules.append(self.datahandler.absorption_spectra[selected_run]['molecule'])
        elif TAB_ENUM[tab_index] == 'iCCD':
            selected_runs, scalings = self.spec_data_dir_tree.getSelectedRuns()
            self.spec_datahandler.load_runs(selected_runs)
            for selected_run in selected_runs:
                autoscalings.append(self.spec_datahandler.dispersed_fluorescence[selected_run].get("autoscaling factor", 1) if self.use_auto_scale_cb.isChecked()
                                    else 1)
                molecules.append(self.spec_datahandler.dispersed_fluorescence[selected_run]['molecule'])
        autoscalings = np.array(autoscalings)
        scalings = np.array(scalings)

        if len(selected_runs) == 0:
            return
        
        # #Get number of sums
        it = QTreeWidgetItemIterator(self.sum_tree)
        n_sums = 0
        while it.value():
            n_sums += 1
            it += 1
        
        # If force sum is checked, sum all spectra without splitting into species
        if self.force_sum_cb.isChecked():
            runs_string = ''
            sum_data_key = ''
            for selected_run in selected_runs:
                runs_string += os.path.basename(selected_run)+' '
                sum_data_key += selected_run+' '
            # Make unique identifier - if the same runs are used with different scalings
            sum_data_key = TAB_ENUM[tab_index]+' Sum {}:'.format(n_sums+1) + sum_data_key
            
            molecule_str = ''
            #Calculate sum spectrum
            if TAB_ENUM[tab_index] == 'PMT':
                sum_data_key = 'PMT '+sum_data_key
                self.datahandler.sum_runs(selected_runs, sum_data_key, 
                                          self.k_calib_cb.isChecked(), self.k_calib_filename.text(),
                                          run_scalings=scalings*autoscalings, run_weights=1/autoscalings)
                molecule_str = self.datahandler.absorption_spectra[sum_data_key]['molecule']
            elif TAB_ENUM[tab_index] == 'iCCD':
                sum_data_key = 'iCCD '+ sum_data_key
                self.spec_datahandler.sum_runs(selected_runs, sum_data_key, run_scalings=scalings*autoscalings, run_weights=1/autoscalings,
                                               bin_width=float(self.sum_bin_width.text().replace(',', '.')) if self.sum_bin_cb.isChecked() else None)
                molecule_str = self.spec_datahandler.dispersed_fluorescence[sum_data_key]['molecule']
            
            #Add sum spectrum to tree
            sum_item = QTreeWidgetItem(self.sum_tree)
            sum_item.setText(self.sum_tree_headers['Sum selection'], 'Sum {} ({})'.format(n_sums+1, TAB_ENUM[tab_index]))
            sum_item.setData(self.sum_tree_headers['Scale'], Qt.ItemDataRole.EditRole, 1)
            sum_item.setText(self.sum_tree_headers['Molecule'], molecule_str)
            sum_item.setText(self.sum_tree_headers['Runs'], runs_string)
            sum_item.setData(self.sum_tree_headers['Runs'], Qt.ItemDataRole.UserRole, sum_data_key)
            sum_item.setCheckState(0, Qt.CheckState.Checked)
            sum_item.setFlags(sum_item.flags() | Qt.ItemFlag.ItemIsEditable)
        else:
            molecules_set = set(molecules)
            for molecule in molecules_set:
                selected_runs_subset = np.array(selected_runs)[np.array(molecules) == molecule]
                scalings_subset = scalings[np.array(molecules) == molecule]
                autoscalings_subset = autoscalings[np.array(molecules) == molecule]


                runs_string = ''
                sum_data_key = ''
                for selected_run in selected_runs_subset:
                    runs_string += os.path.basename(selected_run)+' '
                    sum_data_key += selected_run
                # Make unique identifier - if the same runs are used with different scalings
                sum_data_key = TAB_ENUM[tab_index]+' Sum {}'.format(n_sums+1) + sum_data_key
                
                #Calculate sum spectrum
                if TAB_ENUM[tab_index] == 'PMT':
                    self.datahandler.sum_runs(selected_runs_subset, sum_data_key, 
                                              self.k_calib_cb.isChecked(), self.k_calib_filename.text(),
                                              run_scalings=scalings_subset*autoscalings_subset, 
                                              run_weights=1/autoscalings_subset)
                elif TAB_ENUM[tab_index] == 'iCCD':
                    bin_width = float(self.sum_bin_width.text().replace(',','.')) if self.sum_bin_cb.isChecked() else None
                    self.spec_datahandler.sum_runs(selected_runs_subset, sum_data_key, 
                                                   run_scalings=scalings_subset*autoscalings_subset,
                                                   run_weights=1/autoscalings_subset, bin_width=bin_width)
                
                #Add sum spectrum to tree
                sum_item = QTreeWidgetItem(self.sum_tree)
                sum_item.setText(self.sum_tree_headers['Sum selection'], 'Sum {} ({})'.format(n_sums+1, TAB_ENUM[tab_index]))
                sum_item.setData(self.sum_tree_headers['Scale'], Qt.ItemDataRole.EditRole, 1)
                sum_item.setText(self.sum_tree_headers['Molecule'], molecule)
                sum_item.setText(self.sum_tree_headers['Runs'], runs_string)
                sum_item.setData(self.sum_tree_headers['Runs'], Qt.ItemDataRole.UserRole, sum_data_key)
                sum_item.setCheckState(0, Qt.CheckState.Checked)
                sum_item.setFlags(sum_item.flags() | Qt.ItemFlag.ItemIsEditable)
                n_sums += 1

        #Update display
        self.display_button_clicked()


    def export_button_clicked(self):
        #First loop over selected sums
        it = QTreeWidgetItemIterator(self.sum_tree, 
                                     QTreeWidgetItemIterator.IteratorFlag.Checked | QTreeWidgetItemIterator.IteratorFlag.NoChildren)
        prev_save_dir = 'C:\\Users\\au643642\\OneDrive - Aarhus universitet\\Documents\\PhD\\Projects\\'
        while it.value():
            sum_name = it.value().text(self.sum_tree_headers['Sum selection'])
            scale = it.value().data(self.sum_tree_headers['Scale'], Qt.ItemDataRole.EditRole)
            sum_data_key = it.value().data(self.sum_tree_headers['Runs'], Qt.ItemDataRole.UserRole)
            it += 1
            
            wls = np.nan
            signal = np.nan
            signal_std = np.nan
            molecule_str = ''
            
            if sum_data_key.split()[0] == 'PMT':
                abs_spectrum = self.datahandler.absorption_spectra[sum_data_key]
                wls = abs_spectrum['wavelengths']
                if self.display_laser_calib_cb.isChecked(): wls = calibrate_laser(wls, self.laser_combo_box.currentText())
                signal = abs_spectrum['absorption']
                signal_std = abs_spectrum['absorption_std']
                molecule_str = abs_spectrum['molecule']
            if sum_data_key.split()[0] == 'iCCD':
                disp_spectrum = self.spec_datahandler.dispersed_fluorescence[sum_data_key] 
                wls = disp_spectrum['wavelengths']
                signal = disp_spectrum['fluorescence']
                signal_std = disp_spectrum['fluorescence_std']
                molecule_str = disp_spectrum['molecule']

            # Ask for location and filename
            filename, _ = QFileDialog.getSaveFileName(self, "Save sum", prev_save_dir+molecule_str, filter='ASCII file (*.txt)')
            if len(filename) > 0:
                prev_save_dir = filename[::-1].split('/', maxsplit=1)[1][::-1]+'/'
                #Save spectrum
                header = 'Molecule: '+molecule_str+'\n'
                header += 'Creation date: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header += '\nRuns: \n'
                for file in sum_data_key.split('O:')[1:]:
                    header += 'O:'+file + '\n'
                np.savetxt(filename, np.vstack((wls, signal, signal_std)).T, header=header)
                

    def reset_sum_button_clicked(self):
        self.sum_tree.clear()
    def clear_selection_button_clicked(self):
        #Clear selection (checks) in data directory tree
        self.datadirtree.clearSelection()
        self.spec_data_dir_tree.clearSelection()

        #Clear in sum tree
        it = QTreeWidgetItemIterator(self.sum_tree, 
                QTreeWidgetItemIterator.IteratorFlag.Checked)
        while it.value():
            it.value().setCheckState(0,Qt.CheckState.Unchecked)
            it += 1
            

    def autoscale_button_clicked(self):
        tab_index = self.dir_tabs.currentIndex()
        TAB_ENUM = {0: 'PMT', 1 : 'iCCD'}
        if TAB_ENUM[tab_index] == 'PMT':
            selected_runs, scales = self.datadirtree.getSelectedRuns()
            self.datahandler.auto_rescale_runs(selected_runs, self.k_calib_cb.isChecked(), self.k_calib_filename.text())
        elif TAB_ENUM[tab_index] == 'iCCD':
            selected_runs, scales = self.spec_data_dir_tree.getSelectedRuns()
            self.spec_datahandler.auto_rescale_runs(selected_runs)

        self.use_auto_scale_cb.setChecked(True)
        self.display_button_clicked()
    
    def on_resize(self, event):
        self.update_ticks()

        self.sc.figure.tight_layout()
        self.sc.draw()
        
        super().resizeEvent(event)
    def update_ticks(self):
        plot_width, plot_height = self.sc.get_width_height()
        x_range = self.sc.axes.get_xlim()
        y_range = self.sc.axes.get_ylim()
        n_x_major_ticks = int(plot_width / 100)
        major_ticks_spacings = np.array([0.5, 1,     2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000])
        minor_ticks_spacings = np.array([0.1, 0.2, 0.5, 1,  2,  5,  5, 10,  20,  50,  50, 100,  200,  500, 500,  1000])

        x_ticks_spacing_idx = np.argmin(np.abs(major_ticks_spacings - (x_range[1]-x_range[0])/n_x_major_ticks))
        self.sc.axes.xaxis.set_major_locator(ticker.MultipleLocator(major_ticks_spacings[x_ticks_spacing_idx]))
        self.sc.axes.xaxis.set_minor_locator(ticker.MultipleLocator(minor_ticks_spacings[x_ticks_spacing_idx]))
        self.sc.axes.tick_params(which='major', axis='both', length=10, width=1.2)
        self.sc.axes.tick_params(which='minor', axis='both', length=6, width=1.2)

    def sum_item_double_clicked(self, item : QTreeWidgetItem, column : int):
        flags = item.flags()
        if column == self.sum_tree_headers['Scale'] or column == self.sum_tree_headers['Sum selection']:
            item.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
        else:
            item.setFlags(flags & (~Qt.ItemFlag.ItemIsEditable))
    
    def refresh_trees_button_clicked(self):
        self.datadirtree.reload_folders()
        self.spec_data_dir_tree.reload_folders()

    def keyPressEvent(self, event):
        pass

    # def on_tree_right_clicked(self):
    #     pop_figure = CanvasWidget()
    #     CanvasWidget.show()

    # def auto_display_clicked(self):
    #     if self.auto_display_data_cb.isChecked():
    #         self.datadirtree.tree.itemChanged.connect(self.display_button_clicked)
    #         self.spec_data_dir_tree.tree.itemChanged.connect(self.display_button_clicked)
    #     else:
    #         self.datadirtree.tree.itemChanged.disconnect()
    #         self.spec_data_dir_tree.tree.itemChanged.disconnect()

    def k_calib_open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File", 
            "O:\\Nat_IFA-fullc\\experimental\\expdata\\Fluorescence_LUNA2\\Photo diode calibration\\AbsoluteCalibFiles\\", 
            "Calibration files (*.txt)"
        )
        if filename:
            path = Path(filename)
            self.k_calib_filename.setText(str(path))

    def reload_selected_runs(self):
        selected_runs, scales = self.datadirtree.getSelectedRuns()
        self.datahandler.load_runs(selected_runs, self.k_calib_cb.isChecked(), self.k_calib_filename.text(),
                                   force_reload=True)

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
