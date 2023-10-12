import sqlite3
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator

class LUNADatabase:
    def __init__(self, database_file):
        #TODO: Make a check that the file exists - otherwise create a new database with the given name
        self.db_conn = sqlite3.connect(database_file)
    def __del__(self):
        self.db_conn.close()

    def add_entry(self, ):
        #Check if entry already exists
        pass
        # id_to_check = 1
        # cursor = self.db_conn.execute("SELECT id FROM molecules WHERE id=?", (id_to_check,))
        # if cursor.fetchone() is not None:
        #     print("ID already exists in the database")
        # else:
        #     print("ID does not exist in the database")

    def save_database(self,):
        pass

    #Searches for a molecule in the database and gives the results in the tree 
    #TODO: include partial matches, and search for synonyms as well. R575 is also Rho575 
    def search_for_molecule(self, molecule : str, tree : QTreeWidget):
        tree.clear()
        #Make the PMT branch

        #Make the iCCD branch
        pass


    ## If entries in the db tree are selected, also select them in their respective PMT/iCCD trees
    ## Such the display feature still works as intended, then