from DataDirTree import DataDirTree
import glob, os
import numpy as np
from PyQt6.QtWidgets import QTreeWidgetItem, QTreeWidgetItemIterator
from PyQt6.QtWidgets import QInputDialog

from PyQt6.QtCore import Qt, QSize
from datetime import datetime 
import SpecHeaderReader
import SpecDataHandler

class SpectrometerDirTree(DataDirTree):
    def __init__(self, data_dir_path : str, spec_data_handler : SpecDataHandler):
        self.header_indices = {'Run selection' : 0, 'Scale': 1, 'Molecule': 2, 'Excitation wavelength': 3, 'Filter':4, 'Comments': 5}
        super().__init__(data_dir_path)
        self.spec_data_handler = spec_data_handler
    def reload_folders(self):
        self.tree.clear()
        self.tree.itemDoubleClicked.connect(self.on_double_clicked)
        for year_dir in sorted(glob.glob(self.data_dir_path+'\\*\\'))[::-1]:
            year_item = QTreeWidgetItem(self.tree)
            year_item.setText(0, os.path.basename(year_dir[:-1]))
            
            files = glob.glob(year_dir+'*.dat')
            #Sort files in dates and make date-folders
            dates = []
            for file in files:
                d = os.path.basename(file)[:6]
                dates.append(d[4:6]+'-'+d[2:4]+'-20'+d[:2])
            dates_set = set(dates)
            date_dirs = sorted(dates_set, key=lambda date: datetime.strptime(date, "%d-%m-%Y"), reverse=True)
            for date_dir in date_dirs:
                root_item = QTreeWidgetItem(year_item)  # create a QTreeWidgetItem for the current root folder
                root_item.setFlags(root_item.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                root_item.setText(0, os.path.basename(date_dir))  # set the display text to the name of the current folder
                root_item.setData(0, Qt.ItemDataRole.UserRole, date_dir)  # set the folder path as user data for the item

                dt_date = datetime.strptime(date_dir, "%d-%m-%Y")
                date_string_prefix = str(dt_date.year)[2:]+str(dt_date.month).zfill(2)+str(dt_date.day).zfill(2)

                run_files = glob.glob(year_dir+date_string_prefix+'*.dat')
                run_files.sort(key=lambda run: int(run.strip('\\').split('\\')[-1].split('_')[0][6:]))
                for run_file in run_files:
                    dir_path = os.path.basename(run_file).split('_')[0]
                    run_item = QTreeWidgetItem(root_item)  # create a QTreeWidgetItem for the current subfolder
                    run_item.setCheckState(0, Qt.CheckState.Unchecked)
                    run_item.setFlags(run_item.flags() | Qt.ItemFlag.ItemIsEditable)
                    run_item.setText(0, dir_path)  # set the display text to the name of the current folder
                    run_item.setData(0, Qt.ItemDataRole.UserRole, run_file)  # set the folder path as user data for the item
                    run_item.setData(self.header_indices['Scale'], Qt.ItemDataRole.EditRole, 1)
                    
    def on_double_clicked(self, item, column_no):
        if column_no == 0 and item.childCount() == 0:
            run = item.data(0, Qt.ItemDataRole.UserRole)
            start, ok1 = QInputDialog.getDouble(self.tree, 'Set Removal Interval', 'Start')
            if ok1: stop, ok2 =  QInputDialog.getDouble(self.tree, 'Set Removal Interval', 'End')
            if ok1 and ok2:
                wls = self.spec_data_handler.dispersed_fluorescence[run]['wavelengths']
                idxs_removed = np.logical_and(wls > start, wls < stop)
                self.spec_data_handler.dispersed_fluorescence[run]['fluorescence'] = np.copy(self.spec_data_handler.dispersed_fluorescence[run]['raw_fluorescence'])
                self.spec_data_handler.dispersed_fluorescence[run]['fluorescence_autoscale'] = np.copy(self.spec_data_handler.dispersed_fluorescence[run]['raw_fluorescence'])
                self.spec_data_handler.dispersed_fluorescence[run]['fluorescence'][idxs_removed] = np.nan
                if np.argwhere(idxs_removed)[0][0] == 0 or np.argwhere(idxs_removed)[-1][0] == (len(wls)-1):
                    #Removed from either edge - remove datapoints
                    pass
                else:
                    if np.sum(idxs_removed) > 0:
                        flu = self.spec_data_handler.dispersed_fluorescence[run]['fluorescence']

                        left_idx = np.argwhere(idxs_removed)[0][0]-1
                        right_idx = np.argwhere(idxs_removed)[-1][0]+1
                        n_left = min(left_idx, 3)
                        n_right = min(len(idxs_removed) - right_idx, 3)

                        left_avg  = flu[left_idx-n_left:left_idx]
                        right_avg = flu[right_idx:right_idx+n_right]

                        left_avg = np.mean(left_avg)
                        right_avg = np.mean(right_avg)
                        self.spec_data_handler.dispersed_fluorescence[run]['fluorescence_autoscale'][idxs_removed] = (left_avg+right_avg)/2



    def on_item_expanded(self, item : QTreeWidgetItem):
        # Date folder is expanded
        if not (item.parent() is None) and item.parent().parent() is None:
            date_dir = item.data(0, Qt.ItemDataRole.UserRole)
            n_children = item.childCount()
            for i in range(n_children):
                child = item.child(i)
                run_file = child.data(0, Qt.ItemDataRole.UserRole)
                
                _, header_dict, _ = SpecHeaderReader.read_header_string(run_file)

                child.setText(self.header_indices['Molecule'], header_dict.get('Ion name', ''))
                child.setText(self.header_indices['Excitation wavelength'], header_dict.get('Excitation wavelength', ''))
                child.setText(self.header_indices['Filter'], header_dict.get('Flourescence filter type', '') + ' @ ' + header_dict.get('Filter wavelength', '') + ' nm')
                comments = header_dict.get('[Notes]', '')
                child.setText(self.header_indices['Comments'], comments.split('\n')[0])
                child.setToolTip(self.header_indices['Comments'], comments) 
                
    def getSelectedRuns(self):
        selected_dirs = []
        scales = []
        it = QTreeWidgetItemIterator(self.tree, 
                QTreeWidgetItemIterator.IteratorFlag.Checked | QTreeWidgetItemIterator.IteratorFlag.NoChildren)
        while it.value():
            selected_dirs.append(it.value().data(0, Qt.ItemDataRole.UserRole))
            scales.append(it.value().data(1, Qt.ItemDataRole.EditRole))
            it += 1
        return selected_dirs, scales