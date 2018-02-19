# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FOV_Data_overview_designer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtWidgets
from config.actions import methods as fov
from config.actions import actions as ac
from config import DataOverviewGUI as GUI
import os
import pickle

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class FOVDataOverview(QtWidgets.QMainWindow):

    def __init__(self):
        super(FOVDataOverview, self).__init__()
        self.ui = GUI.Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.parameters()

        # actions
        self.ui.actionClose_2.triggered.connect(self.close_application)
        self.ui.actionOpen_directory.triggered.connect(self.change_directory)
        self.ui.loadData.clicked.connect(self.plot_radiance)
        self.ui.plotFOV.clicked.connect(self.fov_plot)

        # parameters GUI
        self.ui.channelBox.setValue(1)
        self.ui.waveBox.setRange(200, 600)
        self.ui.waveBox.setValue(500)
        self.ui.deltaBox.setValue(12)
        self.ui.directoryText.setText(self.cwd)


    def parameters(self):
        self.cwd = os.getcwd()
        self.config = self.config_read()


    def config_read(self):

        with open(self.cwd + '/config/configuration.pickle', 'rb') as f:
            self.config = pickle.load(f)
        return self.config

    def close_application(self):
        print('Programm closed by user')
        sys.exit()

    def change_directory(self):
        name = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open folder')
        self.cwd = name
        print(self.cwd)

    def load_data(self):
        """Load data specified in the channel"""

        self.config['channel'] = int(self.ui.channelBox.value())
        self.config['wavelength'] = int(self.ui.waveBox.value())
        self.config['delta'] = self.ui.deltaBox.value()
        print(self.config['channel'], self.config['wavelength'])

        self.update_config()
        try:
            position, data = fov.read_data(self.config, self.cwd)
            print(position.shape, data.shape)
            print('Data loaded')

        except:
            position = 0
            data = 0
            print(
                'Data cannot be found, change channel or problem will be closed')

        return position, data


    def update_config(self):
        """Update configuration file in the config directory"""
        with open(self.cwd + '/config/configuration.pickle', 'wb') as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)
            print('config updated', self.cwd)
        return self.config


    def plot_radiance(self):
        """Plot of radiance in function of the position of the lamp"""
        position, data = self.load_data()
        azim, zen, radiance = fov.select_values(data, position, self.config)
        fov.plot_fov(azim, zen, radiance, self.config)


    def fov_plot(self):
        """Show the FOV for """
        position, data = self.load_data()

        if self.ui.azimuthCheckBox.isChecked() == True:
            fov.FOV_plot_azim(data, position, self.config)
        elif self.ui.zenithCheckBox.isChecked() == True:
            fov.FOV_plot_zen(data, position, self.config)
        else:
            print('No direction selected')


if __name__ == "__main__":
    Program = QtWidgets.QApplication(sys.argv)
    MyProg = FOVDataOverview()
    MyProg.show()
    sys.exit(Program.exec_())

