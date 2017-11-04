# Processing

#import matplotlib.pyplot as plt
from functions import functions as fov

config = dict()
config['channel'] = 37
config['wavelength'] = 500
config['delta'] = 15


# run programm

data = fov.read_data()
print('Data loaded, shape:', data.shape)

fov.plot_fov(data, config)

print('listo')
