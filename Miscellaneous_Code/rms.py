import matplotlib.pyplot as plt
import poppy
import numpy as np
from run_zernikePSF import create_coeffs_list, specify_coeffs_list

def calculate_rms(number_of_zerinke_modes, number_of_scans, max_coeff_value, seed):
    # create_coeffs_list(number_of_scans, number_of_zerinke_modes, max_coeff_value, seed)
    coeffs_list = create_coeffs_list(number_of_scans, number_of_zerinke_modes, max_coeff_value, seed)
    # coeffs_list = specify_coeffs_list()
    radius = 2e-3
    wavelength = 1500e-9
    FOV_pixels = 512
    h = 60e-6/2
    f = 4.5e-3 * 2
    theta = np.arctan(h/f) / np.pi * 180 * 60 * 60 # in arcsec
    pixscale = theta/FOV_pixels

    rms_list = []
    peak_to_valley_list = []
    plot = False
    # rms_list.append(poppy.ZernikeWFE(radius=2e-3, coefficients=coeffs).peaktovalley())
    for coeffs in coeffs_list:
        # print(coeffs)
        coeffs = np.asarray(coeffs) * 1e-6
        osys = poppy.OpticalSystem()
        circular_aperture = poppy.CircularAperture(radius=radius)
        osys.add_pupil(circular_aperture)
        thinlens = poppy.ZernikeWFE(radius=radius, coefficients=coeffs)
        osys.add_pupil(thinlens)
        osys.add_detector(pixelscale=pixscale, fov_pixels=FOV_pixels)
        psf_with_zernikewfe, all_wfs = osys.calc_psf(wavelength=wavelength, display_intermediates=False, return_intermediates=True)
        pupil_wf = all_wfs[1] #this one ***
        if plot:
            final_wf = all_wfs[-1] #sometimes referred to as wf

            psf = psf_with_zernikewfe[0].data
            # plt.figure(1)
            # plt.imshow(psf)
            # plt.figure(2)
            # poppy.display_psf(psf_with_zernikewfe, normalize='peak', cmap='viridis', scale='linear', vmin=0, vmax=1)
            # plt.show()

            plt.figure(1, figsize=[9, 3])
            plt.subplot(131)
            plt.imshow(pupil_wf.phase)
            plt.title('Pupil Phase')
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(final_wf.amplitude**2)
            plt.title('PSF intensity')
            plt.colorbar()
            plt.subplot(133)
            plt.imshow(final_wf.phase)
            plt.title('PSF phase')
            plt.colorbar()
            plt.figure(2)
            plt.imshow(psf)
            plt.title('PSF')
            plt.colorbar()
            plt.show()
        # print(np.std(pupil_wf.phase))
        rms_list += [np.std(pupil_wf.phase)]
        peak_to_valley_list += [np.amax(pupil_wf.phase) - np.amin(pupil_wf.phase)]
        # input('')

    # print(len(rms_list))
    # print(max(rms_list), min(rms_list))
    # print(np.average(rms_list))
    return rms_list, peak_to_valley_list

def main():
    dicts = []
    # dicts += [{'name':'BigOne1', 'zernikes':7, 'number_of_scans':1000, 'max_coeff_value':0.2, 'seed':1}]
    # dicts += [{'name':'BigOne2', 'zernikes':7, 'number_of_scans':1200, 'max_coeff_value':0.2, 'seed':2}]
    # dicts += [{'name':'BigOne3', 'zernikes':11, 'number_of_scans':1000, 'max_coeff_value':0.2, 'seed':3}]
    # dicts += [{'name':'BigOne4', 'zernikes':11, 'number_of_scans':1200, 'max_coeff_value':0.2, 'seed':4}]
    # dicts += [{'name':'BigOne5', 'zernikes':16, 'number_of_scans':1250, 'max_coeff_value':0.2, 'seed':5}]
    # dicts += [{'name':'BigOne6', 'zernikes':16, 'number_of_scans':1250, 'max_coeff_value':0.2, 'seed':6}]
    # dicts += [{'name':'BigOne7', 'zernikes':11, 'number_of_scans':2500, 'max_coeff_value':0.5, 'seed':7}]
    # dicts += [{'name':'BigOne8', 'zernikes':11, 'number_of_scans':2500, 'max_coeff_value':0.5, 'seed':8}]
    # dicts += [{'name':'BigOne9', 'zernikes':19, 'number_of_scans':2500, 'max_coeff_value':0.2, 'seed':9}]
    # dicts += [{'name':'BigOne10', 'zernikes':9, 'number_of_scans':1000, 'max_coeff_value':0.12, 'seed':10}] # ** Incorrect - Fionas
    # dicts += [{'name':'BigOne11', 'zernikes':21, 'number_of_scans':2500, 'max_coeff_value':0.2, 'seed':11}]
    # dicts += [{'name':'BigOne12', 'zernikes':19, 'number_of_scans':2500, 'max_coeff_value':0.5, 'seed':12}]
    # dicts += [{'name':'BigOne13', 'zernikes':19, 'number_of_scans':3250, 'max_coeff_value':0.5, 'seed':13}]
    # dicts += [{'name':'BigOne14', 'zernikes':19, 'number_of_scans':3000, 'max_coeff_value':0.2, 'seed':14}]
    # dicts += [{'name':'BigOne15', 'zernikes':19, 'number_of_scans':2500, 'max_coeff_value':0.2, 'seed':15}]
    # dicts += [{'name':'BigOne16', 'zernikes':19, 'number_of_scans':3000, 'max_coeff_value':0.5, 'seed':16}]
    # dicts += [{'name':'BigOne17', 'zernikes':19, 'number_of_scans':3000, 'max_coeff_value':0.5, 'seed':17}]
    # dicts += [{'name':'BigOne18', 'zernikes':19, 'number_of_scans':3000, 'max_coeff_value':0.5, 'seed':18}]
    # dicts += [{'name':'BigOne19', 'zernikes':19, 'number_of_scans':3500, 'max_coeff_value':0.75, 'seed':19}]
    # dicts += [{'name':'BigOne20', 'zernikes':19, 'number_of_scans':7000, 'max_coeff_value':0.75, 'seed':20}]
    # dicts += [{'name':'BigOne21', 'zernikes':11, 'number_of_scans':1500, 'max_coeff_value':0.5, 'seed':21}]
    # dicts += [{'name':'BigOne22', 'zernikes':19, 'number_of_scans':2500, 'max_coeff_value':0.5, 'seed':22}]
    dicts += [{'name':'BigOne23', 'zernikes':8, 'number_of_scans':2500, 'max_coeff_value':0.5, 'seed':23}]
    dicts += [{'name':'BigOne24', 'zernikes':8, 'number_of_scans':350, 'max_coeff_value':0.5, 'seed':24}]
    dicts += [{'name':'BigOne25', 'zernikes':8, 'number_of_scans':1000, 'max_coeff_value':0.5, 'seed':25}]
    dicts += [{'name':'BigOne26', 'zernikes':8, 'number_of_scans':4000, 'max_coeff_value':0.5, 'seed':26}]
    dicts += [{'name':'BigOne27', 'zernikes':8, 'number_of_scans':1000, 'max_coeff_value':0.5, 'seed':27}]
    dicts += [{'name':'BigOne28', 'zernikes':8, 'number_of_scans':16000, 'max_coeff_value':0.5, 'seed':28}]
    dicts += [{'name':'BigOne29', 'zernikes':8, 'number_of_scans':4000, 'max_coeff_value':0.5, 'seed':29}]
    dicts += [{'name':'BigOne30', 'zernikes':8, 'number_of_scans':4000, 'max_coeff_value':0.5, 'seed':30}]
    dicts += [{'name':'BigOne31', 'zernikes':8, 'number_of_scans':1000, 'max_coeff_value':1., 'seed':31}]
    for dict in dicts:
        # create_statistics(dict)
        analytic_rms(dict['name'])
        read_statistics(dict['name'])

def create_statistics(dictionary):
    print('Creating statistics for', dictionary['name'])
    rms_list, peak_to_valley_list = calculate_rms(dictionary['zernikes'], dictionary['number_of_scans'], dictionary['max_coeff_value'], dictionary['seed'])
    dictionary['rms_list'] = rms_list
    dictionary['peak_to_valley_list'] = peak_to_valley_list
    np.savez('/import/tintagel3/snert/david/Neural_Nets/Data/{}/Data_set_statistics.npz'.format(dictionary['name']),
        name=dictionary['name'],
        zernikes=dictionary['zernikes'],
        number_of_scans=dictionary['number_of_scans'],
        max_coeff_value=dictionary['max_coeff_value'],
        seed=dictionary['seed'],
        rms_list=dictionary['rms_list'],
        peak_to_valley_list=dictionary['peak_to_valley_list']
        )

def combine_rms(coeffs):
    coeffs = np.abs(coeffs)
    rms = 0
    rms += coeffs[1]/2
    rms += coeffs[2]/2
    rms += coeffs[3]/np.sqrt(3)
    rms += coeffs[4]/np.sqrt(6)
    rms += coeffs[5]/np.sqrt(6)
    rms += coeffs[6]/np.sqrt(8)
    rms += coeffs[7]/np.sqrt(8)
    # input(rms)
    return rms


def analytic_rms(name):
    dictionary = np.load('/import/tintagel3/snert/david/Neural_Nets/Data/{}/Data_set_statistics.npz'.format(name))
    print('Calculating analytical RMS for', dictionary['name'])

    coeffs_list = create_coeffs_list(dictionary['number_of_scans'], dictionary['zernikes'], dictionary['max_coeff_value'], dictionary['seed'])
    analytic_rms_list = []
    for coeffs in coeffs_list:
        analytic_rms_list += [combine_rms(coeffs)]
    np.savez('/import/tintagel3/snert/david/Neural_Nets/Data/{}/Data_set_statistics.npz'.format(dictionary['name']),
        name=dictionary['name'],
        zernikes=dictionary['zernikes'],
        number_of_scans=dictionary['number_of_scans'],
        max_coeff_value=dictionary['max_coeff_value'],
        seed=dictionary['seed'],
        rms_list=dictionary['rms_list'],
        peak_to_valley_list=dictionary['peak_to_valley_list'],
        analytic_rms_list=analytic_rms_list
        )


def read_statistics(name):
    dictionary = np.load('/import/tintagel3/snert/david/Neural_Nets/Data/{}/Data_set_statistics.npz'.format(name))

    print('RMS Info')
    print('Average:', np.average(dictionary['rms_list']), 'Max:', np.amax(dictionary['rms_list']), 'Min:', np.amin(dictionary['rms_list']), '\n')
    print('Analytic RMS Info')
    print('Average:', np.average(dictionary['analytic_rms_list']), 'Max:', np.amax(dictionary['analytic_rms_list']), 'Min:', np.amin(dictionary['analytic_rms_list']), '\n')
    print('Peak to Valley Info')
    print('Average:', np.average(dictionary['peak_to_valley_list']), 'Max:', np.amax(dictionary['peak_to_valley_list']), 'Min:', np.amin(dictionary['peak_to_valley_list']), '\n')
    print('***********************************\n')
    plt.figure()
    plt.subplot(131)
    plt.hist(dictionary['rms_list'])
    plt.ylabel('Occurances')
    plt.xlabel('RMS')
    plt.title('RMS of {}'.format(dictionary['name']))
    plt.subplot(132)
    plt.hist(dictionary['analytic_rms_list'])
    plt.xlabel('Analytic RMS')
    plt.title('Analytic RMS of {}'.format(dictionary['name']))
    plt.subplot(133)
    plt.hist(dictionary['peak_to_valley_list'])
    # plt.ylabel('Occurances')
    plt.xlabel('Peak to Valley')
    plt.title('Peak to Valley of {}'.format(dictionary['name']))
    plt.savefig('/import/tintagel3/snert/david/Neural_Nets/Data/{}/Statistics.png'.format(name))
    plt.close()

main()
# read_statistics('BigOne1')
