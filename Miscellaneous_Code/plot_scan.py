import numpy as np
import matplotlib.pyplot as plt
import socket
from scipy import stats
import os


def get_filepath():
    computer_name = socket.gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def scan():

    npz_file = np.load('{}/Neural_Nets/Data/Tip_Scan/NN_data.npz'.format(get_filepath()))
    zernike_data = npz_file['zernikes']
    flux_data = npz_file['fluxes']
    print(zernike_data.shape, zernike_data[0].shape)
    print(flux_data.shape, flux_data[0].shape)
    # print(flux_data)
    # input('')

    plt.figure(1)
    plt.scatter(zernike_data[:], flux_data[:, 0])
    plt.xlabel('Tip Value')
    plt.ylabel('Flux')
    plt.title('Flux Through Central Waveguide')
    plt.figure(2)
    plt.scatter(zernike_data[:], np.sum(flux_data, axis=1))
    plt.xlabel('Tip Value')
    plt.ylabel('L1 Sum')
    plt.title('L1 Sum Over Tip Scan [-0.5, 0.5] microns')
    plt.show()


if socket.gethostname() == 'tauron':
    set_gpu(1)

scan()
