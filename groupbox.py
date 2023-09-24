from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys

class App(QWidget):

    def imageOptionClicked(self,state):
        isChecked = bool(state)
        if (isChecked):
            print(self.checkBoxes.checkedButton().text())

    option_checked = pyqtSignal(str)
    def __init__(self):
        QWidget.__init__(self)

        self.setWindowTitle("GroupBox")
        self.textLabel = QLabel('/dev/video0')

        vbox = QVBoxLayout()
        vbox.addWidget(self.textLabel)
#        layout = QGridLayout()
#        self.setLayout(layout)

        self.checkBoxes = QButtonGroup()
        #'image options')

        hbox = QHBoxLayout()
        options = [QCheckBox("Cards"), QCheckBox("Edge"), QCheckBox("Edge Contours"), QCheckBox("Motion")]
        options[0].setChecked(True)
        for i in range(len(options)):
            hbox.addWidget(options[i])
            self.checkBoxes.addButton(options[i], i)
            options[i].stateChanged.connect(self.imageOptionClicked)


        vbox.addLayout(hbox)
        self.setLayout(vbox)
        
app = QApplication(sys.argv)
screen = App()
screen.show()
sys.exit(app.exec_())
