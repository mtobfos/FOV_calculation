# PyQt test programm
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
from config import functions as fov
#from config import methods as mth
import os
import pickle

#app = QtWidgets.QApplication(sys.argv)

#windows = QtWidgets.QWidget()
#windows.setGeometry(50, 50, 500, 300)
#windows.setWindowTitle("Test Qtpy5")
#windows.show()

cwd = os.getcwd()


def config_read():
    with open(cwd + '/config/configuration.pickle', 'rb') as f:
        config = pickle.load(f)
    return config


class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("FOV Data Overview")

        self.config = config_read()
        
        # close programm
        extractAction = QtWidgets.QAction("&Close", self)
        #extractAction.setShortcut("s")
        extractAction.setStatusTip('Close programm')
        extractAction.triggered.connect(self.close_application)

        # change path to main folder
        openDirectory = QtWidgets.QAction("&Open folder", self)
        openDirectory.setStatusTip('Load folder with the data')
        openDirectory.triggered.connect(self.change_directory)
        
        # menu
        self.statusBar()

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)  # include menu in window (on mac)

        # add options to menu
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(openDirectory)
        fileMenu.addAction(extractAction)


        self.home()
        #self.setWindowIcon(QtWidgets.QIcon(''))



    def home(self):
        """home windows of the programm"""
        # add channel to analyse
        chn = QtWidgets.QLabel('Channel', self)
        chn.move(30, 20)

        self.chnEdit = QtWidgets.QSpinBox(self)
        self.chnEdit.move(120, 25)
        self.chnEdit.resize(40, 20)
        self.chnEdit.setRange(0, 145)

        # Wavelength definition
        wvl = QtWidgets.QLabel('Wavelength', self)
        wvl.move(30, 40)

        self.wvlEdit = QtWidgets.QDoubleSpinBox(self)
        self.wvlEdit.setRange(280, 600)
        self.wvlEdit.move(120, 45)
        self.wvlEdit.resize(60, 20)

        # load data to programm
        btn = QtWidgets.QPushButton("Load data", self)
        #btn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        btn.clicked.connect(self.load_data)
        btn.move(100, 100)

        btnchn = QtWidgets.QPushButton("plot radiance", self)
        btnchn.clicked.connect(self.plot_radiance)
        btnchn.move(200, 100)

        self.show()

    def close_application(self):
        print('Programm closed by user')
        sys.exit()


    def load_data(self):
        """Load data specified in the channel"""
        self.config['channel'] = int(self.chnEdit.value())
        self.config['wavelength'] = int(self.wvlEdit.value())

        print(self.config['channel'], self.config['wavelength'])

        self.update_config()

        position, data = fov.read_data(self.config)
        print(position.shape, data.shape)
        print('Data loaded')
        return position, data


    def update_config(self):
        """Update configuration file in the config directory"""
        with open(cwd + '/config/configuration.pickle', 'wb') as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)
            print('config updated')
        return self.config

    def change_directory(self):
        print('directory changed')


    def plot_radiance(self):
        """Plot of radiance in function of the position f¡of the lamp"""
        position, data = self.load_data()
        azim, zen, radiance = fov.select_values(data, position, self.config)
        fov.plot_fov(azim, zen, radiance, self.config)


def run():
    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()
