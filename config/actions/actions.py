# actions for the programm
import sys
import pickle
from PyQt5 import QtCore, QtWidgets
from config.actions import methods as fov
import os

# Methods of the GUI


class Actions:

    def __init__(self):
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
        name = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open folder')
        cwd = name
        print(cwd)

    def load_data(self):
        """Load data specified in the channel"""
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

        if self.azim.isChecked() == True:
            fov.FOV_plot_azim(data, position, self.config)
        elif self.zen.isChecked() == True:
            fov.FOV_plot_zen(data, position, self.config)
        else:
            print('No direction selected')
