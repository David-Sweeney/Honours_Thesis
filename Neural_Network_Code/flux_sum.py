import numpy as np
import socket
datasets = ('BigOne8',)

def get_filepath():
    computer_name = socket.gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def normalise_zernikes(zernike_data):
    if datasets[0] == 'BigOne10':
        zernike_mean = np.mean(zernike_data, axis=0)
        zernike_deviation = np.std(zernike_data, axis=0)
        zernike_data = (zernike_data - zernike_mean)/zernike_deviation
    elif datasets[0] == 'PracticalData9':
        zernike_data = zernike_data * 2.5
    elif datasets[0] == 'BigOne8' or datasets[0] == 'BigOne12':
        zernike_data = zernike_data * 2
    else:
        zernike_data = zernike_data * 5
    return zernike_data

def fun():
    # Pull data in from file
    filename = '{}/Neural_Nets/Data/{}/NN_data.npz'.format(get_filepath(), datasets[0])
    npz_file = np.load(filename)
    zernike_data = npz_file['zernikes'].copy()
    flux_data = npz_file['fluxes'].copy()
    for i in range(1, len(datasets)):
        print('Now trying', datasets[i])
        filename = '{}/Neural_Nets/Data/{}/NN_data.npz'.format(get_filepath(), datasets[i])
        npz_file = np.load(filename)
        zernike_data = np.concatenate((zernike_data, npz_file['zernikes'].copy()))
        flux_data = np.concatenate((flux_data, npz_file['fluxes'].copy()))

    # Normalise training data
    zernike_data = normalise_zernikes(zernike_data)
    # Do this for fluxes
    if datasets[0] == 'PracticalData9':
        flux_data = flux_data - np.amin(flux_data)
    total_intensity = [sum(t) for t in flux_data]
    average_intensity = sum(total_intensity)/len(total_intensity)
    flux_data = flux_data / average_intensity
    return flux_data

flux_data = fun()
print(flux_data.shape)
print('Total:', sum(sum(flux_data)))
total = 0
for i in range(len(flux_data)):
    print(sum(flux_data[i, :]), len(flux_data[i, :]))
    total += sum(flux_data[i, :])
    # input('')
print(total/2500)
