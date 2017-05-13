import numpy as np
import util
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from gui_viewer import GUIViewer

class MainWindow(QMainWindow):

    def __init__(self, opt_engine, parent=None, win_size=450):
        QMainWindow.__init__(self, parent)
        self.widget = QWidget()
        self.opt_engine = opt_engine

        # hbox1 widgets voxel viewer widget
        self.frame = QFrame()
        self.viewerWidget = GUIViewer(self.frame, opt_engine)
        self.viewerWidget.setFixedSize(win_size, win_size)
        viewerBox = QVBoxLayout()
        viewerBox.addWidget(self.viewerWidget)
        self.frame.setLayout(viewerBox)

        # hbox2 widgets
        self.btnSave = QPushButton("Save")
        btnLayout1 = QHBoxLayout()
        btnLayout1.addWidget(self.btnSave)
        btnWidget1 = QWidget()
        btnWidget1.setLayout(btnLayout1)
        btnWidget1.setFixedWidth(win_size)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.frame)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(btnWidget1)

        vbox1 = QVBoxLayout()
        vbox1.addLayout(hbox1)
        vbox1.addLayout(hbox2)

        self.widget.setLayout(vbox1)
        self.setCentralWidget(self.widget)

        mainWidth = self.viewerWidget.width() + 35
        mainHeight = self.viewerWidget.height() + 70
        self.setGeometry(0, 0, mainWidth, mainHeight)
        self.setFixedSize(self.width(), self.height())
        self.btnSave.clicked.connect(self.save_data)
        self.connect(self.opt_engine, SIGNAL('update_voxels'), self.viewerWidget.update_actor)
        self.opt_engine.start()

    def closeEvent(self, event):
        self.opt_engine.quit()
        self.opt_engine.model.sess.close()

    def save_data(self):
        try:
            self.number
        except:
            self.number = 1
        util.save_binvox("./out/{0}.binvox".format(self.number), self.opt_engine.current_shape)
        print "saved {0}.binvox".format(self.number)
        self.number += 1
