from DataDirTree import DataDirTree
import glob, os
from datetime import datetime
from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtCore import Qt
from PMTHeaderReader import read_header_string


class PMTDirTree(DataDirTree):
    def __init__(self, data_dir_path):
        self.header_indices = {'Run selection' : 0, 'Scale': 1, 'Molecule': 2, 'Quick load': 3, 'Injections': 4, 'Comments': 5}
        super().__init__(data_dir_path)
    def reload_folders(self):
        self.tree.clear()
        for year_dir in sorted(glob.glob(self.data_dir_path+'\\*\\'))[::-1]:
            year_item = QTreeWidgetItem(self.tree)
            year_item.setText(0, os.path.basename(year_dir[:-1]))
            print(year_dir)
            date_dirs = glob.glob(year_dir+'\\*\\')

            date_dirs.sort(key=lambda date: datetime.strptime(os.path.basename(date[:-1]), "%d-%m-%Y"))
            for date_dir in date_dirs[::-1]:
                root_item = QTreeWidgetItem(year_item)  # create a QTreeWidgetItem for the current root folder
                root_item.setFlags(root_item.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                root_item.setText(0, os.path.basename(date_dir[:-1]))  # set the display text to the name of the current folder
                root_item.setData(0, Qt.ItemDataRole.UserRole, date_dir)  # set the folder path as user data for the item

                run_dirs = glob.glob(date_dir+'\\*\\')
                run_dirs.sort(key=lambda run: int(run.strip('\\').split('\\')[-1].split('-')[-1]))

                for run_dir in run_dirs:
                    dir_path = os.path.basename(run_dir[:-1])
                    run_item = QTreeWidgetItem(root_item)  # create a QTreeWidgetItem for the current subfolder
                    run_item.setCheckState(0, Qt.CheckState.Unchecked)
                    run_item.setFlags(run_item.flags() | Qt.ItemFlag.ItemIsEditable)
                    run_item.setText(0, dir_path)  # set the display text to the name of the current folder
                    run_item.setData(0, Qt.ItemDataRole.UserRole, dir_path)  # set the folder path as user data for the item
                    run_item.setData(self.header_indices['Scale'], Qt.ItemDataRole.EditRole, 1)

    def on_item_expanded(self, item : QTreeWidgetItem):
        #Check if date_dir - by checking if its parent doesn't have a parent
        if not (item.parent() is None) and item.parent().parent() is None:
            date_dir = item.data(0, Qt.ItemDataRole.UserRole)
            n_children = item.childCount()
            children_to_remove = []
            for i in range(n_children):
                child = item.child(i)
                run_dir = date_dir + child.data(0, Qt.ItemDataRole.UserRole)
                if len(glob.glob(run_dir + '\\*.dat')) < 3:
                    children_to_remove.append(child)
                else:
                    # Check if combined run file exists, if it does load the run info
                    header_dict = {}
                    combined_run_filename = run_dir+'_channelA.dat'
                    if os.path.exists(combined_run_filename):
                        header_length, dict = read_header_string(combined_run_filename)
                        header_dict = dict
                        child.setText(self.header_indices['Quick load'], 'Yes')
                    else:
                        run_files = glob.glob(run_dir+'\\*.dat')
                        if len(run_files) > 0:
                            header_length, dict = read_header_string(run_files[0])
                            header_dict = dict
                        child.setText(self.header_indices['Quick load'], 'No')

                    child.setText(self.header_indices['Molecule'], header_dict.get('Molecule', ''))
                    child.setText(self.header_indices['Injections'], header_dict.get('Traces per scan step', ''))
                    child.setText(self.header_indices['Comments'], header_dict.get('Comments', ''))
            for child in children_to_remove:
                item.removeChild(child)