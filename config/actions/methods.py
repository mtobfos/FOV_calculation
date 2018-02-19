import datetime
import glob
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
from scipy import interpolate
from numba import jit
import pickle

cwd = os.getcwd()
num_points = 2500 # points for interpolation

def save_configuration(config):
    with open('config/configuration.pickle', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)


def add_align():
    """Read alignment file in the configuration folder"""
    try:
        alignment = pd.read_table(cwd + '/config_files/Alignment_Lab_UV_20120822.dat',
        sep='\s+',
        names=['Channel Number', 'Azimuth', 'Zenith', 'pixel1', 'pixel2',
               'pixel3'], skiprows=1)
    except ValueError:
        print("Add alignment file in folder '~./config_files/'. The alignment file\n"
              "must beginning with 'Alignment_' ")

    return alignment


def fiber_alignment(config, ind=0):
    """Function shows the fiber alignment along the sensor pixels"""
    files = sorted(
        glob.glob(cwd + '/data/{:03d}/measured_data/data_*.txt'.format(
            config['channel'])))

    # load data from txt file
    txt = np.genfromtxt(files[ind], delimiter='', skip_header=11)

    align = add_align()

    # extract pixels of alignment
    pixels = align['pixel1'] + config['channel_pixel_adj']

    plt.figure(figsize=(12, 9), dpi=300)

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax1.plot(txt[500, :], '-*')
    ax1.axis([0, 1060, 0, txt[500, :].max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Channel alignment')

    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    # First section
    ax2.plot(txt[500, :], '-*')
    ax2.axis([0, 200, 0, txt[500, :].max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Initial section')

    ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    # final section
    ax3.plot(txt[500, :], '-*')
    ax3.axis([800, 1060, 0, txt[500, :].max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Final section')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def data_structure(config):
    """ Arrange the data into a common data structure for the analysis
    return an array of form [positions, radiance]

    positions = [azimuths, zeniths]
    radiance = [channels, wavelengths]
    """
    files_h5 = sorted(glob.glob(cwd + '/data/{:03d}/arranged_data/*.h5'.format(config['channel'])))

    # Load data into notebook
    key = 'data'
    dataray = np.zeros([113, 992, len(files_h5)])
    print('arranging data...')

    dark_current = dark(config)

    for j in np.arange(len(files_h5)):
        # load a file
        info = loadhdf5file(files_h5[j], key=key)
        dataray[:, :, j] = info[0][key] - dark_current
        print('loading file ', j, 'of ', len(files_h5))

    positions = np.load(cwd + '/data/{:03d}/positions/positions.npy'.format(config['channel']))
    data = dataray

    with h5py.File(cwd + '/data/{:03d}/data.h5'.format(config['channel']), 'w') as dat:
        if not list(dat.items()):
            dat.create_dataset('/data', data=data, dtype='f4')
            dat.create_dataset('/positions', data=positions, dtype='f4')
        else:
            del dat['data']

            dat.create_dataset('/data', data=data, dtype='f4')
            dat.create_dataset('/positions', data=positions, dtype='f4')
            dat['positions'].dims[0].label = 'azimuth, zenith'
            dat['positions'].dims[1].label = 'time'
            dat['data'].dims[0].label = 'channel'
            dat['data'].dims[1].label = 'wavelength'

    print('data saved')


def loadhdf5file(file_h5, key='data'):
    """Read contains of HDF5 file saved with save2hdf5 function"""

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
    return sorted(glob.glob(cwd + '/data/{:03d}/measured_data/*.txt'.format(config['channel'])))


def files_pos_dir(config):
    """load path to position files """
    return sorted(glob.glob(cwd + '/data/{:03d}/positions/positions.txt'.format(config['channel'])))

def raw_position2columns(config):
    """Convert raw file from copy to 2 colummns file"""
    path = cwd + '/data/{:03d}/positions/'.format(config['channel'])
    file = glob.glob(path + 'positions_*.txt')
    raw = np.genfromtxt(file[0], delimiter='')

    positions = raw[:, 2]
    np.savetxt(path + 'positions.txt', positions, fmt='%.2f')


def add_align():
    """Add alignment information"""
    alignment = pd.read_table(cwd + '/config/Alignment_Lab_UV_20120822.dat',
                              sep='\s+',
                              names=['Channel Number', 'Azimuth', 'Zenith', 'pixel1', 'pixel2',
                                     'pixel3'], skiprows=1)
    return alignment

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
    alignment = pd.read_table(cwd + '/config/Alignment_Lab_UV_20120822.dat',
        sep='\s+',
        names=['Channel Number', 'Azimuth', 'Zenith', 'pixel1', 'pixel2',
               'pixel3'], skiprows=1)

    # Create the directory to save the results
    os.makedirs(os.path.dirname(cwd + '/data/{:03d}/arranged_data/'.format(config['channel'])), exist_ok=True)

    config['path_save'] = cwd + '/data/{:03d}/arranged_data/'.format(config['channel'])

    # --------SKYMAP--------------
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
            sky.create_dataset('/skymap', data=skymap, dtype='f4')
        else:
            del sky['skymap']

            sky.create_dataset('/skymap', data=skymap, dtype='f4')
            sky['skymap'].dims[0].label = 'channel'
            sky['skymap'].dims[1].label = 'Azimuth, Zenith'

    # Save MUDIS file information
    for fil in np.arange(init_file, fin_file, step):

        # Import the data from the file
        fili = np.genfromtxt(files[fil], delimiter='', skip_header=11)

        # ------------RADIANCE DATA RAW---------------
        # create the radiance matrix
        data = np.zeros([113, 992])

        for i in np.arange(113):
            if np.isnan(alignment.iloc[i][3]) == True:
                data[i] = np.nan
            else:
                try:
                    data[i] = fili[:, int(alignment.iloc[i][3] + config['channel_pixel_adj'])]  #
                except:
                    pass

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
            with h5py.File(config['path_save'] + new_name + '.h5', 'w') as datos:

                if not list(datos.items()):
                    # Create two datasets(use only one time)
                    datos.create_dataset('/data', data=data, dtype=np.float32)
                    datos.create_dataset('/skymap', data=skymap, dtype=np.float32)
                else:
                    del datos['data']
                    del datos['skymap']
                    print('data deleted and corrected')
                    datos.create_dataset('/data', data=data, dtype=np.float32)
                    datos.create_dataset('/skymap', data=skymap, dtype=np.float32)

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

                chn = np.arange(1, 114)
                datos.create_dataset('/channel', data=chn, dtype=np.float32)

                datos['data'].dims.create_scale(datos['channel'], 'channel')
                datos['data'].dims[0].attach_scale(datos['channel'])
                datos['data'].dims[0].label = 'channel'
                datos['data'].dims[1].label = 'wavelength'

                datos['skymap'].dims[0].label = 'channel'
                datos['skymap'].dims[1].label = 'Azimuth, Zenith'
                datos.close()

        else:
            print('Exposure are not same', expo, exposure)
            break
        print('File ' + str(fil - init_file + 1) + ' of ' +
              str((fin_file - init_file)) + ' saved')

    print('Completed')


def dark(config):
    """Calculate the dark current in the measurements """

    # -----------DARK CURRENT----------------------
    files = sorted(glob.glob(cwd + '/data/{:03d}/measured_data/dark/dark_*.txt'.format(config['channel'])))
    print(len(files))
    # Import the data from the file
    dark_file = np.genfromtxt(files[0], delimiter='', skip_header=11)

    # Create array to save data
    dark = np.zeros(list(dark_file.shape))

    print('Calculating...')
    cnt = 0
    # Calculate mean of dark files
    for i in np.arange(len(files)):
        dark += np.genfromtxt(files[i], delimiter='', skip_header=11)
        cnt += 1
    dark = dark / (cnt + 1)

    # create the radiance matrix
    dark_current = np.zeros([113, 992])
    alignment = add_align()

    for i in np.arange(113):
        if np.isnan(alignment.iloc[i][3]) == True:
            dark_current[i] = np.nan
        else:
            try:
                dark_current[i] = dark[:, int(alignment.iloc[i][3]) +
                                      config['channel_pixel_adj']]
            except:
                pass

    print('Complete')

    return dark_current


def position_arrange(file, config):
    """ convert the positions txt files in hdf5 """
    raw = np.genfromtxt(file)

    position = np.zeros([int(len(raw) / 2), 2])
    for i in np.arange(int(len(raw) / 2)):
        position[i] = raw[2*i], raw[2*i + 1]

    np.save('data/{:03d}/positions/positions.npy'.format(config['channel']), position)
    print(position)


def read_data(config, cwd):
    """Import FOV data saved in HDF5 files. Data must have the structure
    [azimuth, zenith, radiance] """

    with h5py.File(cwd + '/data/{:03d}/data.h5'.format(config['channel']), 'r') as dat:
        data = dat['data'][:]
        positions = dat['positions'][:]

    return positions, data


def select_values(data, positions, config):
    """Select the values from the data matrix"""

    radianc = data[config['channel'], config['wavelength'], :]
    radiance = radianc / radianc.max()
    zen = positions[:, 1]
    azim = positions[:, 0]

    # correct azimuth values to normal projection
    azim = (azim - config['meas_azim']) * zen / 90  + config['meas_azim']

    return azim, zen, radiance


def plot_fov(azim, zen, radiance, config, add_points=False):
    """Plot the FOV measured in a contour plot """

    delta = config['delta']
    name = 'corrected_FOV'

    ylim_min = config['meas_zen'] - delta
    ylim_max = config['meas_zen'] + delta

    # plot
    cmap = 'nipy_spectral'
    levels = np.arange(0, radiance.max(), 1/30)

    plt.tricontourf(azim, zen, radiance, cmap=cmap, levels=levels)
    plt.ylim(ylim_min, ylim_max)
    plt.title('FOV AZ{} ZN{}'.format(config['meas_azim'], config['meas_zen']))
    plt.xlabel('azimuth angle[deg]', fontsize=12)
    plt.ylabel('zenith angle[deg]', fontsize=12)
    plt.gca().invert_yaxis() #invert y axis

    # plot center of fiber
    plt.plot([-delta + config['meas_azim'], delta - 1 + config['meas_azim']], [config['meas_zen'], config['meas_zen']], '--r')
    plt.plot([config['meas_azim'], config['meas_azim']], [ylim_min, ylim_max], '--r')
    plt.axis(ratio=1)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.axis([-delta + config['meas_azim'], config['meas_azim'] + delta, ylim_max, ylim_min])
    plt.colorbar()

    if add_points == True:
        # add measurement points
        plt.scatter(azim, zen, cmap=cmap, s=0.4, c='w')
        # show centre of FOV
        plt.scatter(config['azimuth_avg'], config['zenith_avg'], cmap=cmap, s=50)

    os.makedirs(
        os.path.dirname(cwd + '/results/'), exist_ok=True)

    plt.savefig('results/{}.png'.format(name), dpi=300)
    plt.show()
    plt.close()


def plot_surf(data, positions, config, azimuth=0, zenith=30):
    """ Plot the FOV into a surface plot """
    azim, zen, radiance = select_values(data, positions, config)

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cmap = 'nipy_spectral'
    ax.plot_trisurf(azim, zen, radiance, cmap=cmap)
    ax.set_title('3D FOV Channel {}, {}nm'.format(config['channel'], config['wavelength']), fontsize=14)
    ax.set_xlabel('azimuth[1]', fontsize=12)
    ax.set_ylabel('zenith[1]', fontsize=12)
    ax.set_zlabel('normalized radiance', fontsize=12)
    plt.gca().invert_yaxis() #invert y axis
    ax.view_init(zenith, azimuth)

    ax.scatter(azim, zen, radiance, s=2)
    plt.show()


# %%%%%%%%%%%%%%%% DATA ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%

def FOV(function, first, last, tol=0.01, value=0.5):
    """ Calcule the FOV of a function with two minimum values and one maximum"""
    val = []
    num = int((last - first) / tol)

    for i in np.linspace(first, last, num=num):
        if (value - tol) <= function(i) <= (value + tol):
            val = np.append(val, i)
        else:
            pass

    # Separate range to determine the max and min value in the degree axis
    fov_min = []
    fov_max = []

    for i in range(len(val)):
        if val[i] < ((first + last) / 2):
            fov_min = np.append(fov_min, val[i])
        else:
            fov_max = np.append(fov_max, val[i])

    fov = np.mean(fov_max) - np.mean(fov_min)
    pos = [np.mean(fov_min), np.mean(fov_max)]

    return fov, pos


def FOV_smoothing(data, positions, config):
    """ Smooth the data to find the maximum of the FOV in noisy data"""
    azim, zen, radiance = select_values(data, positions, config)

    # Set up a regular grid of interpolation point
    azim_new, zen_new = np.linspace(azim.min(), azim.max(),
                             num=num_points), \
                 np.linspace(zen.min(), zen.max(), num=num_points)

    azim_new, zen_new = np.meshgrid(azim_new, zen_new)

    zi = interpolate.griddata((azim, zen), radiance, (azim_new, zen_new),
                                    method='cubic')

    return azim_new, zen_new, zi


def find_centre_fov(data, positions, config):
    """ Find the FOV using the center of mass"""
    azim, zen, radiance = select_values(data, positions, config)
    peak_azim = (azim * radiance).sum() / radiance.sum()
    peak_zen = (zen * radiance).sum() / radiance.sum()

    return peak_azim, peak_zen


def FOV_plot_azim(data, positions, config, show=True):
    """Calculate and plot the FOV for the azimuth plane"""

    # find the maximum of radiance
    peak_azim, peak_zen = find_centre_fov(data, positions, config)

    azimuth, zen, radiance = select_values(data, positions, config)
    # Look for maximum value in the FOV azimuth profile
    indx = []
    tol = 0.25

    for i in np.arange(len(zen)):
        if (peak_zen - tol) <= zen[i] < (peak_zen + tol):
            indx.append(i)

    ind = np.asarray(indx)

    # normalize radiance
    rad_max = data[config['channel'], config['wavelength'], ind[0]:ind[-1] + 1] / \
            data[config['channel'], config['wavelength'], ind[0]:ind[-1] + 1].max()

    # look for azimuth angles in the maximum of radiance plane
    azim = np.zeros(len(ind))
    i = 0
    for val in ind:
        azim[i] = azimuth[val]
        i += 1
    azim = sorted(azim)
    azim = np.asarray(azim, dtype='f4')

    # interpolate data
    rad_az_interp = interpolate.interp1d(azim, rad_max, kind='cubic')
    azim_new = sorted(np.linspace(azim[0], azim[-1], num=num_points))

    if len(radiance) % 2 == 0:
        windows_len = len(radiance) + 1
    else:
        windows_len = len(radiance)

    # smooth the radiance curve
    sm = savgol_filter(rad_az_interp(azim_new), window_length=windows_len, polyorder=5)
    sm_inter = interpolate.interp1d(azim_new, sm, kind='cubic')

    # find the values equal to 0.5
    fov_val, ind_f = FOV(sm_inter, azim_new[0], azim_new[-1], tol=0.01)

    # FOV maximum found
    max_fov = ((ind_f[0] + ind_f[1]) / 2)

    # plot data
    title = 'Angular Response channel {} azimuth, {}nm'.format(config['channel'], config['wavelength'])
    fig, ax = plt.subplots()
    ax.plot(azim, rad_max, 'b*-', label='radiance')
    ax.set_title(title)
    ax.set_xlabel('Azimuth Angle[1]')
    ax.set_ylabel('Normalized Radiance')

    ax.plot([config['meas_azim'], config['meas_azim']], [0, 1], 'r--')
    ax.plot([max_fov, max_fov], [0, 1], 'g--')
    #ax.plot(azim_new, rad_az_interp(azim_new), 'b-', label='smooth radiance')
    ax.plot([ind_f[0], ind_f[1]], [0.5, 0.5])
    ax.plot(azim_new, sm, 'r-', label='smoothing')
    ax.legend()
    textstr = 'FOV={:.2f}º\nFOV pos={:.2f}°\nDeviation={:.2f}°'.format(fov_val,
                                                    max_fov,
                                                    max_fov - config['meas_azim'])
    # place a text box in upper left in axes coords
    ax.text(0.75, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')
    plt.savefig('results/{}.png'.format(title), dpi=300)

    if show == True:
        plt.show()
        plt.close()
    else:
        plt.close()

    return fov_val, max_fov

def FOV_plot_zen(data, positions, config, show=True):
    """Calculate and plot the FOV for the azimuth plane"""

    # find the maximum of radiance
    peak_azim, peak_zen = find_centre_fov(data, positions, config)

    azimuth, zenith, radiance = select_values(data, positions, config)
    # Look for maximum value in the FOV azimuth profile
    indx = []
    tol = 0.1

    for i in np.arange(len(azimuth)):
        if (peak_azim - tol) <= azimuth[i] < (peak_azim + tol):
            indx.append(i)
    ind = np.asarray(indx, dtype='int')

    # normalize measured radiance
    rad_max = data[config['channel'], config['wavelength'], ind] / \
            data[config['channel'], config['wavelength'], ind].max()

    zen = np.zeros(len(ind))
    i = 0
    for val in ind:
        zen[i] = zenith[val]
        i += 1
    zen = sorted(zen)
    zen = np.asarray(zen, dtype='f4')
    # interpolate data
    rad_az_interp = interpolate.interp1d(zen, rad_max, kind='cubic') # interp1d
    zen_new = sorted(np.linspace(zen[0], zen[-1], num=num_points))

    if len(radiance) % 2 == 0:
        windows_len = len(radiance) + 1
    else:
        windows_len = len(radiance)

    # smooth the curve
    sm = savgol_filter(rad_az_interp(zen_new), window_length=windows_len, polyorder=3)
    sm_inter = interpolate.interp1d(zen_new, sm, kind='cubic')

    # find the values equal to 0.5
    fov_val, ind_f = FOV(sm_inter, zen_new[0], zen_new[-1], tol=0.01)

    # FOV maximum found
    max_fov = (ind_f[0] + ind_f[1]) / 2

    # plot data
    title = 'Angular Response channel {} zenith, {}nm'.format(config['channel'], config['wavelength'])
    fig, ax = plt.subplots()
    ax.plot(zen, rad_max, 'b*-', label='radiance')
    ax.set_title(title)
    ax.set_xlabel('Zenith Angle[1]')
    ax.set_ylabel('Normalized Radiance')
    ax.plot([config['meas_zen'], config['meas_zen']], [0, 1], 'r--')
    ax.plot([max_fov, max_fov], [0, 1], 'g--')
    #ax.plot(zen_new, rad_az_interp(zen_new), 'b-', label='smooth radiance')
    ax.plot([ind_f[0], ind_f[1]], [0.5, 0.5])
    ax.plot(zen_new, sm, 'r-', label='smoothing')
    ax.axis([config['meas_zen'] - config['delta'], config['meas_zen'] + config['delta'], 0, 1.1])
    ax.legend()
    textstr = 'FOV={:.2f}º\nFOV pos={:.2f}°\nDeviation={:.2f}°'.format(
        fov_val,
        max_fov,
        max_fov - config['meas_zen'])
    # place a text box in upper left in axes coords
    ax.text(0.75, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')
    plt.savefig('results/{}.png'.format(title), dpi=300)

    if show == True:
        plt.show()
        plt.close()
    else:
        plt.close()

    return fov_val, max_fov


def FOV_avg(data, positions, config, init_wave=400, end_wave=500, step=50, axis='azimuth', show=False):
    """ Find the average FOV point"""

    results = []

    for i in np.arange(init_wave, end_wave, step):
        config['wavelength'] = i
        if axis == 'azimuth':
            fov, pos = FOV_plot_azim(data, positions, config, show=show)
        elif axis == 'zenith':
            fov, pos = FOV_plot_zen(data, positions, config, show=show)
        else:
            print('¡¡¡ Axis specification is not correct, please write azimuth or zenith\n'
                  'in axis parameter !!!')
        results.append((i, fov, pos))

    r = np.asfarray(results)

    avg = r[:, 2].mean()

    config['{}_avg'.format(axis)] = avg
    print('Mean centre FOV in {}: {:.2f}º'.format(axis, avg))

    # plotting FOV and position of azimuth
    fig, ax1 = plt.subplots()

    ax1.plot(r[:, 0], r[:, 1], 'b*-', label='FOV')
    ax1.set_title('Results')
    ax1.set_xlabel('wavelength[pixels]')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("FOV", color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(r[:, 0], r[:, 2], 'r*-', label='position {}'.format(axis))
    ax2.set_ylabel('position {}'.format(axis), color='r')
    ax2.tick_params('y', colors='r')

    plt.show()
    plt.close()
