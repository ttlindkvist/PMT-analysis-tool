import sys, os, glob
from datetime import datetime
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator, QStyledItemDelegate, QLineEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator

from PMTHeaderReader import read_header_string

class DataDirTree:
    header_indices = {'Run selection' : 0, 'Scale': 1, 'Molecule': 2, 'Quick load': 3, 'Comments': 4}
    def __init__(self, data_dir_path):
        self.tree = QTreeWidget()
        self.tree.setMouseTracking(True)
        self.tree.setColumnCount(len(self.header_indices.keys()))
        self.tree.setHeaderLabels(self.header_indices.keys())
        self.tree.clicked.connect(self.clicked)
        self.tree.itemDoubleClicked.connect(self.double_clicked)
        self.tree.setItemDelegateForColumn(self.header_indices['Scale'], EditableDelegate(self.tree))
        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 40)
        self.tree.itemExpanded.connect(self.on_item_expanded)
        self.data_dir_path = data_dir_path
        self.reload_folders()

    
    def reload_folders(self):
        pass

    def on_item_expanded(self, item : QTreeWidgetItem):
        pass

    def clicked(self):
        pass

    def double_clicked(self, item : QTreeWidgetItem, column : int):
        flags = item.flags()
        if column == self.header_indices['Scale']:
            item.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
        else:
            item.setFlags(flags & (~Qt.ItemFlag.ItemIsEditable))

    def getSelectedRuns(self):
        selected_dirs = []
        scales = []
        it = QTreeWidgetItemIterator(self.tree, 
                QTreeWidgetItemIterator.IteratorFlag.Checked | QTreeWidgetItemIterator.IteratorFlag.NoChildren)
        while it.value():
            selected_dirs.append(it.value().parent().data(0, Qt.ItemDataRole.UserRole) +
                 it.value().data(0, Qt.ItemDataRole.UserRole))
            scales.append(it.value().data(self.header_indices['Scale'], Qt.ItemDataRole.EditRole))
            it += 1
        return selected_dirs, scales
    
    def clearSelection(self):
        it = QTreeWidgetItemIterator(self.tree, 
                QTreeWidgetItemIterator.IteratorFlag.Checked)
        while it.value():
            it.value().setCheckState(0,Qt.CheckState.Unchecked)
            it += 1

class EditableDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def createEditor(self, parent, option, index):
        # return the editor widget for the delegate
        editor = QLineEdit(parent)
        editor.setValidator(FloatValidator()) # set the validator to check for float
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