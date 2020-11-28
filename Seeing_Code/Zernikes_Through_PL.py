import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras as K
from socket import gethostname
from math import sqrt, cos, sin, radians, pi
import poppy
import os
plt.ion()

model = None
plane_wave_sum = None
# emulator_time

def get_filepath():
    computer_name = gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def load_model():
    global model
    global plane_wave_sum
    model_name = 'Hestia5_model-f50'
    filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name.split('_')[0], model_name)
    model = K.models.load_model(filepath)
    plane_wave_fluxes = model.predict(np.array([[0.]*7]))
    plane_wave_sum = np.sum(plane_wave_fluxes)
    print('Plane wave sum:', plane_wave_sum)
    return filepath, model_name

def load_from_file(file_name):
    filepath = f'{get_filepath()}/opticspy/{file_name}'
    npz_file = np.load(filepath)
    zernikes = npz_file['zernikes']
    seeings = npz_file['seeings']
    if 'fluxes' in npz_file.keys():
        fluxes = npz_file['fluxes']
        return zernikes, seeings, fluxes
    return zernikes, seeings

def pass_zernike_to_model(zernike):
    # Zernikes for Hestia5 need to be doubled
    zernike = zernike * 2
    return model.predict(np.array([zernike])) / plane_wave_sum

def generate_psf(coeffs):
    coeffs = np.insert(coeffs, 0, 0)
    # Declare physical constants
    radius = 2e-3
    wavelength = 1500e-9
    FOV_pixels = 512 #Increase this (finer) 1st.
    h = 60e-6/2 #Increase this (wider) 2nd
    f = 4.5e-3 * 2
    theta = np.arctan(h/f) / np.pi * 180 * 60 * 60 # in arcsec
    pixscale = theta/FOV_pixels #somehow resize this after - bilinear reduction of resolution

    coeffs = (np.asarray(coeffs)/2) * 1e-6
    # Create PSF
    osys = poppy.OpticalSystem()
    circular_aperture = poppy.CircularAperture(radius=radius)
    osys.add_pupil(circular_aperture)
    thinlens = poppy.ZernikeWFE(radius=radius, coefficients=coeffs)
    osys.add_pupil(thinlens)
    osys.add_detector(pixelscale=pixscale, fov_pixels=FOV_pixels)
    psf_with_zernikewfe, all_wfs = osys.calc_psf(wavelength=wavelength, display_intermediates=False, return_intermediates=True)
    pupil_wf = all_wfs[1] #this one ***
    final_wf = all_wfs[-1] #sometimes referred to as wf
    # psf = psf_with_zernikewfe[0].data
    return pupil_wf.phase, final_wf.amplitude**2

def plot_lantern_outputs(flux):
    plt.figure(1)
    ax_modes = plt.subplot(111, aspect='equal')
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    colour_map = 'plasma'
    ax_modes.set_facecolor('black') # Set it to the 0 of the colour map
    mid_edge = sqrt(3)

    # Define the coordinates of the centred hexagonal grid
    x_coords = [0, cos(radians(-120)), cos(radians(-60)), 1, cos(radians(60)), cos(radians(120)), -1,
        mid_edge*cos(radians(-150)), 2*cos(radians(-120)), 0,
        2*cos(radians(-60)), mid_edge*cos(radians(-30)), 2, mid_edge*cos(radians(30)),
        2*cos(radians(60)), 0, 2*cos(radians(120)), mid_edge*cos(radians(150)), -2]
    y_coords = [0, sin(radians(-120)), sin(radians(-60)), 0, sin(radians(60)), sin(radians(120)), 0,
        mid_edge*sin(radians(-150)), 2*sin(radians(-120)), mid_edge*sin(radians(-90)),
        2*sin(radians(-60)), mid_edge*sin(radians(-30)), 0, mid_edge*sin(radians(30)),
        2*sin(radians(60)), mid_edge*sin(radians(90)), 2*sin(radians(120)), mid_edge*sin(radians(150)), 0]

    # Set limits slightly larger than the grid so that dots are not cut in half
    plt.xlim([-2.6, 2.6])
    plt.ylim([-2.6, 2.6])

    vmax=0.35
    for x, y, c in zip(x_coords, y_coords, flux):
        ax_modes.add_patch(matplotlib.patches.Circle((x, y), radius=0.5, facecolor=cm.plasma(c/vmax), edgecolor=None))
    plt.title('Photonic Lantern Outputs')

    cax_modes = make_axes_locatable(ax_modes).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=vmax),
                cmap=colour_map), cax=cax_modes).ax.set_ylabel('Normalised Flux')
    plt.show()
    plt.pause(0.01)

def main():
    if gethostname() == 'tauron':
        set_gpu(1)
    load_model()
    for file in [f'seeing_10min_{i}.npz' for i in range(20)]:
        zernikes, seeings = load_from_file(file)
        # zernikes, seeings, fluxes = load_from_file()
        fluxes = []
        psfs = []
        for i in range(len(zernikes)):
            print(f'{i}/{len(zernikes)} completed.')
            flux = pass_zernike_to_model(zernikes[i])[0]
            _, psf = generate_psf(zernikes[i])
            # plot_lantern_outputs(flux)
            # plt.figure(2)
            # plt.imshow(seeings[i])
            # plt.show()
            # plt.pause(0.001)
            fluxes += [flux]
            psfs += [psf]
        np.savez(f'{get_filepath()}/opticspy/{file}', seeings=np.array(seeings), zernikes=np.array(zernikes), fluxes=np.array(fluxes), psfs=np.array(psfs))



if __name__ == '__main__':
    main()
