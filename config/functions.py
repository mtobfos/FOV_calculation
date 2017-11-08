import datetime
import glob
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd



def data_structure(config):
    """ Arrange the data into a common data structure for the analysis"""
    cwd = os.getcwd()
    files_h5 = sorted(glob.glob(cwd + '/data/{:03d}/arranged_data/*.h5'.format(config['channel'])))

    # Load data into notebook
    key = 'data'
    dataray = np.zeros([113, 992, len(files_h5)])
    time_meas = []

    for j in np.arange(0, len(files_h5)):
        # load a file
        info = loadhdf5file(files_h5[j], key=key)
        dataray[:, :, j] = info[0][key]

    positions = np.load(cwd + '/data/{:03d}/positions.npy'.format(config['channel']))
    data = np.array([positions, dataray])
    np.save('data/{:03d}/data.npy'.format(config['channel']), data)


def loadhdf5file(file_h5, key='data'):
    """Read contains of HDF5 file saved with dat2hdf5_mudis function"""

    with h5py.File(file_h5, 'r') as data:
        # Add datasets to dictionary
        info_value = {}
        info_attrs = {}

        for i in np.arange(len(data.items())):
            info_value.update({str(list(data.items())[i][0]): data[str(list(data.items())[i][0])].value})

        for i in np.arange(len(data[key].attrs)):
            info_attrs.update({list(data[key].attrs.keys())[i]: list(data[key].attrs.values())[i]})

    return info_value, info_attrs


def files_data_dir(config):
    """Add the raw file directory"""
    cwd = os.getcwd()
    files = sorted(glob.glob(cwd + '/data/{:03d}/measured_data/*.txt'.format(config['channel'])))
    return files


def files_pos_dir(config):
    """ """
    cwd = os.getcwd()
    return sorted(glob.glob(cwd + '/data/{:03d}/positions/*.txt'.format(config['channel'])))


def save2hdf5(files, config, init_file=0, fin_file=3, step=1, expo='700'):

    """Function to convert raw data from MUDIS .txt file to hdf5 file with
    attributes.
    Parameters
    ----------
    files:
    init_file:
    fin_file:
    expo:

     """
    alignment = pd.read_table(
        'config/Alignment_Lab_UV_20120822.dat',
        sep='\s+',
        names=['Channel Number', 'Azimuth', 'Zenith', 'pixel', 'pixel',
               'pixel'], skiprows=1)

    # Create the directory to save the results
    os.makedirs(os.path.dirname('data/{:03d}/arranged_data/'.format(config['channel'])), exist_ok=True)

    config['path_save'] = 'data/{:03d}/arranged_data/'.format(config['channel'])

    #--------SKYMAP--------------
    # Create the directory to save the results
    os.makedirs(
        os.path.dirname(config['path_save'] + 'calibration_files/'), exist_ok=True)

    # Extract skymap from alignment file
    skymap = np.zeros([len(alignment), 2])

    for i in np.arange(len(skymap)):
        skymap[i] = alignment['Azimuth'][i], alignment['Zenith'][i]

    # Save Skymap information
    with h5py.File(config['path_save'] + 'calibration_files/skymap_radiance.h5', 'w') as sky:

        if not list(sky.items()):
            sky.create_dataset('/skymap', data=skymap)
        else:
            del sky['skymap']

            sky.create_dataset('/skymap', data=skymap)
            sky['skymap'].attrs['Columns'] = 'Azimuth, Zenith'

    # Save MUDIS file information
    for fil in np.arange(init_file, fin_file, step):

        # Import the data from the file
        fili = np.genfromtxt(files[fil], delimiter='', skip_header=11)

        # ------------RADIANCE DATA RAW---------------
        # create the radiance matrix
        data = np.zeros([113, 992])

        for i in np.arange(113):
            if str(alignment.iloc[i][3]) == 'nan':
                data[i] = np.nan
            else:
                data[i] = fili[:, int(
                    alignment.iloc[i][3] + config['channel_pixel_adj'])]  #
                # read the pixels index
                # in the aligment file and copy the
                # data in the radiance matrix']))

        # Correct time for the file UTC
        name = os.path.split(files[fil])

        # Read name of the file (correct time)
        time = name[1][6:25]
        # convert time to datetime format
        time = datetime.datetime.strptime(time, '%d.%m.%Y_%H_%M_%S')
        # print(time)
        new_name = datetime.datetime.strftime(time, format='%Y%m%d_%H%M%S')

        with open(files[fil], 'r') as fili:
            dat = fili.readlines()

        # Extract information from .dat file
        exposure = int(dat[4][12:-1])
        NumAve = int(dat[7][17:-1])
        CCDTemp = int(dat[8][15:-1])
        NumSingMes = int(dat[10][27:-1])
        ElectrTemp = int(dat[9][23:-1])

        # Create the directory to save the results
        os.makedirs(os.path.dirname(config['path_save']),
                    exist_ok=True)

        if int(exposure) == int(expo):
            # Create a file in the disk
            datos = h5py.File(config['path_save'] + new_name + '.h5', 'w')

            if not list(datos.items()):
                # Create two datasets(use only one time)
                datos.create_dataset('/data', data=data)
                datos.create_dataset('/skymap', data=skymap)
            else:
                del datos['data']
                del datos['skymap']
                print('data deleted and corrected')
                datos.create_dataset('/data', data=data)
                datos.create_dataset('/skymap', data=skymap)

            # Add attributes to datasets
            datos['data'].attrs['time'] = str(time)
            datos['data'].attrs['Exposure'] = exposure
            datos['data'].attrs['NumAver'] = NumAve
            datos['data'].attrs['CCDTemp'] = CCDTemp
            datos['data'].attrs['NumSingMes'] = NumSingMes
            datos['data'].attrs['ElectrTemp'] = ElectrTemp
            datos['data'].attrs['Latitude'] = '52.39N'
            datos['data'].attrs['Longitude'] = '9.7E'
            datos['data'].attrs['Altitude'] = '65 AMSL'

            datos['skymap'].attrs['Columns'] = 'Azimuth, Zenith'

            datos.close()

        else:
            print('Exposure are not same', expo, exposure)
            break
        print('File ' + str(fil - init_file + 1) + ' of ' +
              str((fin_file - init_file)) + ' saved')

    print('Completed')


def position_arrange(file, config):
    """ convert the positions txt files in hdf5 """
    raw = np.genfromtxt(file)

    position = np.zeros([int(len(raw) / 2), 2])
    for i in np.arange(int(len(raw) / 2)):
        position[i] = raw[2*i], raw[2*i + 1]

    np.save('data/{:03d}/positions.npy'.format(config['channel']), position)

    print(position)


def read_data(config):
    """Import FOV data saved in npy files. data muss have the structure
    [azimuth, zenith, radiance] """
    data = np.load('data/{:03d}/data.npy'.format(config['channel']))
    return data


def select_values(data, config):
    """Select the values from the data matrix"""

    radianc = data[1][config['channel'], config['wavelength'], :]
    radiance = radianc / radianc.max()
    zen = data[0][:, 1]
    azim = data[0][:, 0]

    return azim, zen, radiance


def plot_fov(data, config, add_points=False):
    """Plot the FOV measured in a contour plot """

    delta = config['delta']
    name = 'corrected_FOV'

    azim, zen, radiance = select_values(data, config)

    # parameter of plot
    zeni_fib = 12 # zen_fib
    azim_fib = 0 # azim_fib - 180

    ylim_min = 0 # zeni_fib - delta
    ylim_max = zeni_fib + delta

    # plot
    cmap = 'nipy_spectral'
    levels = np.arange(0, radiance.max(), 1/30)

    plt.tricontourf(azim, zen, radiance, cmap=cmap, levels=levels)
    plt.ylim(ylim_min, ylim_max)
    plt.title('FOV AZ{} ZN{}'.format(azim_fib, zeni_fib))
    plt.xlabel('azimuth angle[deg]', fontsize=12)
    plt.ylabel('zenith angle[deg]', fontsize=12)
    plt.gca().invert_yaxis() #invert y axis

    # plot center of fiber
    plt.plot([-delta + azim_fib, delta - 1 + azim_fib], [zeni_fib, zeni_fib], '--r')
    plt.plot([azim_fib, azim_fib], [ylim_min, ylim_max], '--r')

    plt.axis(ratio=1)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.axis([-13, 13, ylim_max, ylim_min])
    plt.colorbar()

    if add_points == True:
        # add measurement points
        plt.scatter(azim, zen, cmap=cmap, s=0.4, c='w')

    plt.savefig('results/{}.png'.format(name), dpi=300)
    plt.show()
    plt.close()

def plot_surf(data, config):

    azim, zen, radiance = select_values(data, config)

    x = np.cos(azim) * np.sin(zen)
    y = np.sin(azim) * np.sin(zen)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_trisurf(x, y, radiance,
                    cmap='nipy_spectral', edgecolor='none');
    ax.set_ylim([-0.2, 0.5])
    ax.set_xlim([-0.2, 0.5])

    plt.show()

# def save_data(data, config):
#     """ save the selected data into a txt file """
#     azim, zen, radiance = select_values(data, config)
#     datei = np.zeros([len(zen), 3])
#
#     for i in np.arange(len(zen)):
#         datei[i] = azim[i], zen[i], radiance[i]
#
#     datos = pd.DataFrame(datei,
#                          columns=['azimuth', 'zenith', 'radiance'])
#
#     print(datos.head())
#
#     datos.to_excel('datei')
