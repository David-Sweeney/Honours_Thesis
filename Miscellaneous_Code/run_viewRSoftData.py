from viewRSoftData import *
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA
import os
# os.nice(-10)
plt.ion()


def extract_from_RSoft(scan_name, datapath, scan_set):
    r = Rsoftdata(datapath)
    r.readMulti(scan_set, keepAllResults=True, showPlots=True)
    r.saveForNN(datapath + 'NN_data.npz')

def delete_and_compress_data(scan_name, datapath, scan_set):
    # Delete inputfield.fld files
    files_to_delete = [i for i in os.listdir(datapath) if '_inputfield.fld' in i]
    for file in files_to_delete:
        os.remove(datapath + file)

    # Zip the remaining files, excluding the NN_data.npz file
    with ZipFile(datapath + 'raw_data.zip', 'w', compression=ZIP_DEFLATED) as zip_object:
    # with ZipFile(datapath + 'raw_data.zip', 'w', compression=ZIP_BZIP2) as zip_object:
    # with ZipFile(datapath + 'raw_data.zip', 'w', compression=ZIP_LZMA) as zip_object:
        count = 1
        files_to_zip = [i for i in os.listdir(datapath) if i.startswith(scan_set)] + ['runAllBatfiles.bat'] # + ['allResults.npz']
        length = len(files_to_zip)
        for file in files_to_zip:
            if count % 10 == 0:
                print('File {} out of {}.'.format(count, length))
            count += 1
            zip_object.write(datapath + file, file)
            os.remove(datapath + file)

def decompress(scan_name, datapath, scan_set):
    with ZipFile(datapath + 'raw_data.zip', 'r', compression=ZIP_DEFLATED) as zip_object:
        print('Got here!')
        zip_object.extractall(datapath)
    os.remove(datapath + 'raw_data.zip')

def compare():
    scan_name = 'Hestia5-f38_Rotations2'
    datapath = '/import/tintagel3/snert/david/Neural_Nets/Data/{}/'.format(scan_name)
    scan_set = 'seven_zernike_'

    true_waveguides = np.load(datapath + 'NN_data.npz')['fluxes'].tolist()
    scan_name = 'Hestia5-f38_Rotations3'
    datapath = '/import/tintagel3/snert/david/Neural_Nets/Data/{}/'.format(scan_name)
    # extract_from_RSoft(scan_name, datapath, scan_set)
    npz_file = np.load(datapath + 'NN_data.npz')
    print(npz_file['zernikes'])
    print(npz_file['fluxes'])
    waveguides = npz_file['fluxes'].tolist()

    for i in range(len(true_waveguides)):
        for _ in range(i):
            true_waveguides[i].insert(6, true_waveguides[i].pop(1))
            # Outer hexagon rotates all round by 2
            true_waveguides[i].insert(18, true_waveguides[i].pop(7))
            true_waveguides[i].insert(18, true_waveguides[i].pop(7))


            waveguides[i].insert(6, waveguides[i].pop(1))
            # Outer hexagon rotates all round by 2
            waveguides[i].insert(18, waveguides[i].pop(7))
            waveguides[i].insert(18, waveguides[i].pop(7))
    # print(waveguides)

    deviation = waveguides-np.average(true_waveguides, axis=0)
    print('Coarse Data')
    print('MAE:', np.average(abs(deviation)))
    print('MSE:', np.average(np.square(deviation)))

    deviation = true_waveguides-np.average(true_waveguides, axis=0)
    print('\nFine Data')
    print('MAE:', np.average(abs(deviation)))
    print('MSE:', np.average(np.square(deviation)))

    deviation = waveguides-np.average(waveguides, axis=0)
    print('\nCoarse on Coarse Data')
    print('MAE:', np.average(abs(deviation)))
    print('MSE:', np.average(np.square(deviation)))
    print('MAE between coarse and fine average:', np.average(abs(np.average(true_waveguides, axis=0) - np.average(waveguides, axis=0))))

    print('\nL1 sums')
    print('Coarse Data:', np.sum(waveguides))
    print('Fine Data:', np.sum(true_waveguides))
    print('Difference:', np.sum(true_waveguides)-np.sum(waveguides), 100*(np.sum(true_waveguides)-np.sum(waveguides))/np.sum(true_waveguides))

    print('Model performance on these examples:')

if __name__ == '__main__':
    compare()
    # Define path
    # scan_name = 'BigOne40'
    scan_name = 'Random_8_Zernike01'
    datapath = '/import/tintagel3/snert/david/Neural_Nets/Data/{}/'.format(scan_name)
    scan_set = 'seven_zernike_'

    # extract_from_RSoft(scan_name, datapath, scan_set)

    # delete_and_compress_data(scan_name, datapath, scan_set)

    # decompress(scan_name, datapath, scan_set)

    # for scan in ['BigOne25', 'BigOne26', 'BigOne27', 'BigOne28', 'BigOne29', 'BigOne30']:
    #     datapath = '/import/tintagel3/snert/david/Neural_Nets/Data/{}/'.format(scan)
    #     delete_and_compress_data(scan, datapath, scan_set)

# r = Rsoftdata('/import/tintagel3/snert/david/Reverse_Injection/doublerun1/')
# r.readall('bptmp')
# r.plotall()
# input('Stop here.')

# datapath = os.path.dirname(os.path.realpath(__file__)) + '/scan2/'

# r = Rsoftdata(datapath)
# r.readall()
# r.readall('seven_zenike_zernikePSFs_0.0000_-0.0717')
# r.plotall()
# input('Press any key to continue')
# r.readall('seven_zenike_zernikePSFs_-0.1718_-0.1706')
# r.plotall()
# input('Press any key to continue')


# Save Data
# r.readMulti(scan_set, keepAllResults=False, showPlots=False)
# r.saveAllResults(datapath + 'allResults.npz')
# 685e-9
# r.saveForNN(datapath + 'NN_data.npz')
# npz_file = np.load(datapath + 'NN_data.npz')
# zernike_data = npz_file['zernikes'].copy()
# flux_data = npz_file['fluxes'].copy()
# for i in range(len(zernike_data)):
#     for j in range(len(zernike_data[i])):
#         zernike_data[i][j] = zernike_data[i][j] / (685e-3/(2*np.pi))
# np.savez(datapath + 'NN_data.npz', zernikes=zernike_data, fluxes=flux_data)




# Load data
# r.loadResults(showPlots=False)
# r.saveForNN(datapath + 'NN_data.npz')

# r.readMulti(scanset, keepAllResults=False)
# r.readMulti(scanset, keepAllResults=False, readOne=10)
# r.plotall()

# print('Final fluxes:')
# print(r.finalFluxes())

#r.saveAllResults()

# Load data instead:
# r.loadResults('results_WLscan02.npz', showPlots=False)
# r.loadResults(showPlots=True)

# r.plotFinalFluxes()

# nfiles = 101
# wlList = np.linspace(1.45, 1.65, nfiles)
# r.plotFinalFluxes(xvals=wlList, norm=None)
# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Flux / TotalFlux')
# input('Press any key to continue')


# for k in range(r.MONdata.shape[1]):
#     plt.plot(r.MONdata[:, k])
#     plt.ylim([0, 1])
#     print(k)
#     plt.pause(0.5)

# # Test effect of averaging final points in monitors
# plt.figure()
# fluxes = []
# for k in range(1, 4000, 10):
#     fluxes.append(r.finalFluxes(useNPts=k))
# p=plt.plot(fluxes)

#np.savez('testsave.npz', allResults=r.allResults)
