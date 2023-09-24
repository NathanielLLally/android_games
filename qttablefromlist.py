# Import necessary modules
import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QDesktopWidget

# Declare a dictionary variable with marks
marks = {'CSE-401': [78, 93, 67, 88, 78],
         'CSE-404': [90, 59, 82, 73, 89],
         'CSE-406': [81, 80, 74, 83, 67],
         'CSE-407': [81, 80, 98, 83, 72]}


class TableFromList(QTableWidget):
    def __init__(self, data, *args):
        # Call parent constructor
        QTableWidget.__init__(self, *args)

        # Declare a list of the student IDS
        self.ID_list = ['0189945', '0154590', '0196734', '0185611', '0178956']
        # Set the necessary configurations fot the table
        self.data = data
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setColumnWidth(0, 100)
        for i in range(4):
            self.setColumnWidth(i, 80)
        self.setMinimumWidth(400)
        self.setWindowTitle("Mark Sheet")

        # Declare the variable to set the header content
        headers = []
        headers.append('')
        # for loop to read the keys of the dictionary
        for n, key in enumerate(sorted(self.data.keys())):
            headers.append(key)
            # for loop to read the values of the dictionary
            for m, item in enumerate(self.data[key]):
                ID = QTableWidgetItem(self.ID_list[m])
                self.setItem(m, 0, ID)
                newVal = QTableWidgetItem(str(item))
                self.setItem(m, n+1, newVal)

        # Set the header label of the table
        self.setHorizontalHeaderLabels(headers)

        # Set the tooltips for the headers
        self.horizontalHeaderItem(1).setToolTip("Multimedia ")
        self.horizontalHeaderItem(2).setToolTip("Artificial Intelligent")
        self.horizontalHeaderItem(3).setToolTip("Advanced Database")
        self.horizontalHeaderItem(4).setToolTip("Unix Programming")

        # Read the particular cell value
        self.clicked.connect(self.on_click)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # Display the window in the center of the screen
        win = self.frameGeometry()
        pos = QDesktopWidget().availableGeometry().center()
        win.moveCenter(pos)
        self.move(win.topLeft())
        self.show()

    def on_click(self):
        for ItemVal in self.selectedItems():
            # Read the header value based on the selected cell
            subject = self.horizontalHeaderItem(ItemVal.column()).text()
            # Print the detail information of the mark
            print("\n", self.ID_list[ItemVal.row()], " got ", ItemVal.text(), " in ", subject)

# Create app object and execute the app
app = QApplication(sys.argv)
table = TableFromList(marks, 5, 5)
table.show()
app.exec()
