import sys, os, glob
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator, QStyledItemDelegate, QLineEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator


class DataDirTree:
    def __init__(self):
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(['Run selection', 'scale'])
        self.tree.clicked.connect(self.clicked)
        self.tree.setItemDelegateForColumn(1, EditableDelegate(self.tree))
        
        data_dir_path = "E:\\LUNA2\\PMTTestData"
        for year_dir in sorted(glob.glob(data_dir_path+'\\*\\'), key=os.path.getmtime):
            year_item = QTreeWidgetItem(self.tree)
            year_item.setText(0, os.path.basename(year_dir[:-1]))
            
            for date_dir in sorted(glob.glob(year_dir+'\\*\\'), key=os.path.getmtime):
                root_item = QTreeWidgetItem(year_item)  # create a QTreeWidgetItem for the current root folder
                root_item.setFlags(root_item.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                root_item.setText(0, os.path.basename(date_dir[:-1]))  # set the display text to the name of the current folder
                root_item.setData(0, Qt.ItemDataRole.UserRole, date_dir)  # set the folder path as user data for the item
            
                for dir_name in sorted(glob.glob(date_dir+'\\*\\'),key=os.path.getmtime):
                    dir_path = os.path.basename(dir_name[:-1])
                    dir_item = QTreeWidgetItem(root_item)  # create a QTreeWidgetItem for the current subfolder
                    dir_item.setCheckState(0, Qt.CheckState.Unchecked)
                    dir_item.setText(0, dir_path)  # set the display text to the name of the current folder
                    dir_item.setData(0, Qt.ItemDataRole.UserRole, dir_path)  # set the folder path as user data for the item
                    dir_item.setData(1, Qt.ItemDataRole.EditRole, 1)
                    dir_item.setFlags(dir_item.flags() | Qt.ItemFlag.ItemIsEditable)

        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 40)
    
    def clicked(self):
        pass
        # for selected_run in self.getSelectedRuns():
        #     print(selected_run)
    def getSelectedRuns(self):
        selected_dirs = []
        scales = []
        it = QTreeWidgetItemIterator(self.tree, 
                QTreeWidgetItemIterator.IteratorFlag.Checked | QTreeWidgetItemIterator.IteratorFlag.NoChildren)
        while it.value():
            selected_dirs.append(it.value().parent().data(0, Qt.ItemDataRole.UserRole) +
                 it.value().data(0, Qt.ItemDataRole.UserRole))
            scales.append(it.value().data(1, Qt.ItemDataRole.EditRole))
            it += 1
        return selected_dirs, scales

class EditableDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def createEditor(self, parent, option, index):
        # return the editor widget for the delegate
        editor = QLineEdit(parent)
        editor.setValidator(FloatValidator()) # set the validator to check for integers
        return editor
        
    def setEditorData(self, editor, index):
        # set the text in the editor to the current text in the item
        text = index.model().data(index, role=Qt.ItemDataRole.DisplayRole)
        editor.setText(str(text))
        
    def setModelData(self, editor, model, index):
        # update the item text with the text in the editor
        text = editor.text()
        model.setData(index, float(text), role=Qt.ItemDataRole.EditRole)
        

class FloatValidator(QValidator):
    def validate(self, input_string, pos):
        # check if the input string can be converted to an integer
        try:
            float(input_string)
            return QValidator.State.Acceptable, input_string, pos
        except ValueError:
            return QValidator.State.Invalid, input_string, pos