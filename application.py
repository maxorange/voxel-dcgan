import sys
import qdarkstyle
import config
from PyQt4.QtGui import QApplication
from ui import MainWindow
from opt import ConstrainedOpt
from model import DCGANTest

if __name__ == "__main__":
    model = DCGANTest(config.nz, config.nsf, config.nvx, config.batch_size)
    model.restore(config.params_path)
    opt_engine = ConstrainedOpt(model)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
    window = MainWindow(opt_engine)
    window.setWindowTitle("voxel-dcgan")
    window.show()
    window.viewerWidget.interactor.Initialize()
    sys.exit(app.exec_())
