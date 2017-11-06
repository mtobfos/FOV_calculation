import matplotlib.pyplot as plt
import numpy as np


def read_data():
    """Import FOV data saved in npy files. data muss have the structure
    [azimuth, zenith, radiance] """

    data = np.load('data/data.npy')
    return data


def plot_fov(data, config, add_points=False):
    """Plot the FOV measured in a contour plot """

    delta = config['delta']
    name = 'corrected_FOV'

    radianc = data[2][config['channel'], config['wavelength'], :]
    radiance = radianc / radianc.max()
    zen = data[1]
    azim = data[0]

    # parameter of plot
    zeni_fib = 12 # zen_fib
    azim_fib = 0 # azim_fib - 180

    ylim_min = 0 # zeni_fib - delta
    ylim_max = zeni_fib + delta

    # plot
    cmap = 'nipy_spectral'
    levels = np.arange(0, radiance.max(), 1/30)

    #fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

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
    plt.axes().set_aspect('equal')
    plt.axis([-13, 13, ylim_max, ylim_min])
    plt.colorbar()

    if add_points == True:
        # add measurement points
        plt.scatter(azim, zen, cmap=cmap, s=0.4, c='w')

    plt.savefig('results/{}.png'.format(name), dpi=300)
    plt.show()
    plt.close()
