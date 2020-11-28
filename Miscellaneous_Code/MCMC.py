# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
from tensorflow import keras as K
# import emcee
import poppy
from scipy import stats
# import cProfile
# import pstats
import os
from scipy.optimize import basinhopping, minimize, fmin_l_bfgs_b
from math import sin, cos, radians, sqrt, pi

### Currently set up for +-0.5 Zernike Models

model = None
plane_wave_sum = None

class SomeResult:
    def __init__(self, x, function):
        self.x = x
        self.fun = function

def get_filepath():
    computer_name = socket.gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def load_model(zernikes):
    global model
    global plane_wave_sum
    if zernikes == 7:
        model_name = 'Simple_model-b57'
        # model_name = 'Simple_model-b21'
    elif zernikes == 8:
        model_name = 'Hestia5_model-f50'
        # model_name = 'Hestia10_model-b18'
        # model_name = 'Hestia10_model-sp27'
    elif zernikes == 9:
        model_name = 'Hyperion5_model-b9'
    elif zernikes == 10:
        model_name = 'Athena_model-b58'
    elif zernikes == 11:
        # model_name = 'Apollo_model-b1'
        # model_name = 'Geryon_model-b1'
        model_name = 'Apollo5_model-b11'
    elif zernikes == 16:
        model_name = 'Orpheus_model-d3'
    else:
        raise ValueError('Number of Zernikes Modes of {} is not supported.'.format(zernikes))
    filepath = '{}/Neural_Nets/Models/{}/{}.h5'.format(get_filepath(), model_name.split('_')[0], model_name)
    model = K.models.load_model(filepath, compile=False)
    model.compile(optimizer=K.optimizers.Adam(), loss='mse')
    # plane_wave_fluxes = model.predict(np.array([[0.]*(zernikes-1)]))
    plane_wave_fluxes = get_fluxes(np.array([0.]*(zernikes-1)), calibrate=True)
    plane_wave_sum = np.sum(plane_wave_fluxes)
    print('Plane wave sum:', plane_wave_sum)
    return filepath, model_name

def get_fluxes(current_values, calibrate=False):
    # mode_values = current_values
    # current_values = np.zeros(42)
    # current_values[2] = mode_values[0]
    # current_values[9] = mode_values[1]
    # current_values[20] = mode_values[2]
    # current_values[25] = mode_values[3]
    # current_values[26] = mode_values[4]
    # current_values[35] = mode_values[5]
    # current_values[40] = mode_values[6]
    # current_values[41] = mode_values[7]

    # current_values = np.zeros(7)
    # current_values[2] = mode_values[2]

    if calibrate:
        return model.predict(np.array([current_values]))
    return model.predict(np.array([current_values])) / plane_wave_sum

def function_L2(current_values):
    L2 = np.sum(np.square(get_fluxes(current_values)))
    return -L2

def function_amax(current_values):
    return -np.amax(get_fluxes(current_values))

def function_maximise_single(current_values, args):
    return -get_fluxes(current_values)[0][args]

def function_funnel_x(current_values, args=3):
    sorted_fluxes = sorted(get_fluxes(current_values)[0], reverse=True)
    return -np.sum(sorted_fluxes[:args])

def function_nulling_central(current_values):
    fluxes = get_fluxes(current_values)
    L1 = np.sum(fluxes)
    return fluxes[0][0] + (1 - L1)

def function_amin(current_values):
    fluxes = get_fluxes(current_values)
    L1 = np.sum(fluxes)
    return np.amin(fluxes) + (1 - L1)

def function_nulling_x(current_values, args=3):
    fluxes = get_fluxes(current_values)
    L1 = np.sum(fluxes)
    sorted_fluxes = sorted(fluxes[0], reverse=False)
    return np.sum(sorted_fluxes[:args]) + (1 - L1)

def function_specified(current_values, args):
    # args is a numpy array which is the goal fluxes
    fluxes = get_fluxes(current_values)
    distance_from_goal = args - fluxes
    return np.sum(np.square(distance_from_goal))

def rosenbrock(current_values):
    a = 1
    b = 100
    return np.exp(-((a-current_values[0])**2 + b*(current_values[1]-current_values[0]**2)**2))

def metropolis_step(current_values, prop, current_func_value, acceptance):
    for mode_value in prop:
        if abs(mode_value) > 1:
            raise ValueError('Proposed values out of bounds', prop)
    prop_func_value = function(prop)
    r = (prop_func_value/current_func_value)
    r = min(1, r)
    # r = (rosenbrock(prop)/rosenbrock(current_values))

    if np.random.rand() < r:
        acceptance += 1
        current_values = prop
        current_func_value = prop_func_value
    return current_values, current_func_value, acceptance

def MCMC(zernikes):
    current_values = np.random.rand(zernikes-1)*2-1
    current_values[2:] = np.zeros((zernikes-3))
    best = function(current_values)
    current = function(current_values)
    count = 0
    iterations = 0
    acceptance = 0
    max_iterations = 1000
    history = np.zeros((max_iterations+1, zernikes-1))
    history[iterations] = np.array([current_values])
    while iterations < max_iterations:
        # prop = np.add(current_values, (np.random.rand(zernikes-1)*2-1)/1000)
        prop = np.zeros(len(current_values))
        prop[0] = current_values[0]+np.random.normal()/3
        prop[1] = current_values[1]+np.random.normal()/3
        out_of_bounds = False
        for mode in prop:
            if abs(mode) > 1:
                out_of_bounds = True
        if out_of_bounds:
            continue
        if iterations % 100 == 0:
            print(iterations)
        iterations += 1
        current_values, current, acceptance = metropolis_step(current_values, prop, current, acceptance)
        history[iterations] = np.array(current_values)
        if current > best:
            best = current
            count = 0
        else:
            count += 1
    print('Acceptance rate:', acceptance/max_iterations)
    return history, best

def scan(zernikes):
    scan_range = 7
    x_vals = np.linspace(-1, 1, scan_range)
    y_vals = np.linspace(-1, 1, scan_range)
    z_vals = np.linspace(-1, 1, scan_range)
    # x, y = np.meshgrid(x_vals, y_vals)
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals)
    # results = np.zeros((scan_range, scan_range))
    results = np.zeros((scan_range, scan_range, scan_range))
    # print(np.array([[x_vals[0], y_vals[0]] + [0]*(zernikes-3)]))
    # input('')
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            for k in range(len(z_vals)):
            # results[i, j] = function([x_vals[i], y_vals[j]] + [0]*(zernikes-3))
                results[i, j, k] = function([x_vals[i], y_vals[j], z_vals[k]] + [0]*(zernikes-4))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(x, y, results, cmap=cm.coolwarm)
    surf = ax.scatter(x, y, z, c=np.ravel(results), cmap=cm.coolwarm)
    fig.colorbar(surf)
    ax.set_xlabel('Tip')
    ax.set_ylabel('Tilt')
    # ax.set_zlabel('Defocus')
    plt.show()

    # Single variable plots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_vals, np.average(results, axis=(1, 2)))
    plt.xlabel('Tip')
    plt.ylabel('exp(-L2)')

    plt.subplot(3, 1, 2)
    plt.plot(x_vals, np.average(results, axis=(0, 2)))
    plt.xlabel('Tilt')
    plt.ylabel('exp(-L2)')

    plt.subplot(3, 1, 3)
    plt.plot(x_vals, np.average(results, axis=(0, 1)))
    plt.xlabel('Defocus')
    plt.ylabel('exp(-L2)')

    plt.tight_layout(pad=1.2)
    plt.show()

    # np.savez('{}/MCMC/scan_L1_zernikes_{}.npz'.format(get_filepath(), zernikes), results=results, x=x, y=y)

def run_own(zernikes):
    history, best = MCMC(zernikes)
    save_and_plot(history)

def save_and_plot(history):
    num = 12
    np.savez('{}/MCMC/mcmc_run_{}.npz'.format(get_filepath(), num), history=history)
    history = history[:,:2]
    # print(history)
    # KDE Stuff
    kde = stats.kde.gaussian_kde(history.T)

    # Regular grid to evaluate kde upon
    x_flat = np.r_[-1:1:128j]
    y_flat = np.r_[-1:1:128j]
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

    z = kde(grid_coords.T)
    z = np.reshape(z, x.shape)

    plt.figure()
    plt.imshow(np.flip(z, axis=0), extent=[-1, 1, -1, 1])
    plt.xlabel("Tip value")
    plt.ylabel("Tilt value")
    # plt.plot(history[:,0], history[:,1], 'k.')
    plt.colorbar()
    plt.savefig('{}/MCMC/mcmc_own_kde_{}.png'.format(get_filepath(), num))
    plt.show()

    plt.figure()
    # np.savez('/import/silo4/snert/david/MCMC/banana.npz', history=history)
    plt.hist2d(history[:, 0]/2.5, history[:, 1]/2.5, 10)
    plt.xlabel("Tilt value (LRM units)")
    # plt.xlabel("x-value")
    plt.ylabel("Tip value (LRM units)")
    # plt.ylabel("y-value")
    plt.colorbar()
    plt.scatter(history[0, 0]/2.5, history[0, 1]/2.5, color='r', marker='x')
    plt.xlim([-0.4, 0.4])
    plt.ylim([-0.4, 0.4])
    # plt.gca().set_yticks([]);
    plt.savefig('{}/MCMC/mcmc_own_run_{}.png'.format(get_filepath(), num))
    # plt.imshow(z)
    # plt.savefig('/import/silo4/snert/david/MCMC/banana.png')
    plt.show()

def run_emcee(zernikes):
    n_dim, n_walkers = zernikes - 1, 16
    initial_values = np.random.rand(n_walkers, zernikes-1)*2-1
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, function)
    sampler.run_mcmc(initial_values, 1000)
    np.savez('{}/mcmc_run.npz'.format(get_filepath()), samples=sampler.get_chain())
    samples = sampler.get_chain(flat=True)/5
    print(samples.shape)
    plt.hist(samples[:, 0], 100, color="k", histtype="step")
    plt.xlabel("Tilt value")
    plt.ylabel("Occurances")
    # plt.gca().set_yticks([]);
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    plt.savefig('{}/mcmc_run.png'.format(get_filepath()))

def callback_info(x, f, accept):
    print('*****************')
    # print('Loc:', x[:2], 'Value:', f)
    print('Loc:', x, '\nValue:', f)
    print('*****************')

def step_take(x):
    # print('Step take!')
    out_of_bounds = True
    while out_of_bounds:
        # prop = np.zeros(len(x))
        # prop[0] = x[0]+np.random.normal()/3
        # prop[1] = x[1]+np.random.normal()/3
        prop = x + np.random.normal(size=x.shape)/2
        out_of_bounds = False
        for mode in prop:
            # if abs(mode) > 1:
            if mode > -0.2 or mode < -0.35:
                out_of_bounds = True
                break
    return prop

def accept_check(f_new, x_new, f_old, x_old):
    out_of_bounds = False
    for mode in x_new:
        if abs(mode) > 1:
            out_of_bounds = True
            break
    return not out_of_bounds

def run_gd(zernikes, exploration, num=None, nit=100):
    current_values = np.random.rand(zernikes-1)*2-1
    # current_values[3:] = np.zeros((zernikes-4))
    # current_values /= 10
    print(current_values)
    # bounds = [(-1, 1)]*(zernikes-1)
    bounds = [(-0.35, -0.2)]*(zernikes-1)
    # bounds = [(-1, 1)]*(3) + [(0, 0)]*(zernikes-4)
    print('Bounds:', bounds)
    minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bounds, "tol":0, "options":{"eps":0.0001}}
    if exploration == 'amax' : function = function_amax
    elif exploration == 'L2' : function = function_L2
    elif exploration == 'funnelx':
        function = function_funnel_x
        minimizer_kwargs['args'] = num
    elif exploration == 'amin' : function = function_amin
    elif exploration == 'nulling_central' : function = function_nulling_central
    elif exploration == 'nullingx':
        function = function_nulling_x
        minimizer_kwargs['args'] = num
    elif exploration == 'maximise':
        function = function_maximise_single
        minimizer_kwargs['args'] = num
    elif exploration == 'specified':
        function = function_specified
        minimizer_kwargs['args'] = num
    else:
        print('Unrecognised exploration', exploration)
        input('')
    if num is None:
        print('Commencing', exploration)
    else:
        print('Commencing', exploration, num)
    result = basinhopping(function, current_values, nit, minimizer_kwargs=minimizer_kwargs, take_step=step_take, disp=False)
    # result = basinhopping(function, current_values, 100, minimizer_kwargs=minimizer_kwargs, callback=callback_info, accept_test=accept_check, disp=True)
    # result = fmin_l_bfgs_b(function, current_values, approx_grad=True, bounds=bounds, pgtol=0, epsilon=0.0001)
    print(result)
    return result

def main(zernikes):
        load_model(zernikes)
        # scan(zernikes)
        run_gd(zernikes)
        # print(model.predict(np.array([[ 0.60606425,  0.02293867, -0.16446762,  0.07364395, -0.11932309,
        # 0.00207138, -0.12089375]]))/plane_wave_sum)
        # print(np.sum(model.predict(np.array([[ 0.60606425,  0.02293867, -0.16446762,  0.07364395, -0.11932309,
        # 0.00207138, -0.12089375]]))/plane_wave_sum))

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

def generate_summary(result, exploration, subplot, pretty=False):
    outputs = get_fluxes(result.x)[0]
    pupil_phase, psf_intensity = generate_psf(result.x)
    # Text summary
        # Type of optimization
        # Function value
        # Array of inputs
        # Array of outputs
        # Exploration specific stats?

    plt.subplot(3, 4, 4*subplot-3)
    plt.axis('off')
    if pretty:
        plt.text(0.2, 0.3, exploration, fontsize=24)
    else:
        plt.title(exploration)
        inputs_text = 'Inputs: {}\n{}'.format([round(i, 4) for i in result.x[:4]], [round(i, 4) for i in result.x[4:]])
        outputs_text = 'Outputs: {}\n{}\n{}\n{}'.format([round(i, 4) for i in outputs[:5]], [round(i, 4) for i in outputs[5:10]],
            [round(i, 4) for i in outputs[10:15]], [round(i, 4) for i in outputs[15:]])
        plt.text(0, 0.5, 'Value: {}\n{}\n{}\nL1 Sum: {}'.format(
        result.fun, inputs_text, outputs_text, np.sum(outputs)))

    # Pupil Phase
    ax_pupil_phase = plt.subplot(3, 4, 4*subplot-2)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    SC = plt.imshow(pupil_phase, cmap='twilight', aspect='equal', vmin=-pi, vmax=pi)
    plt.title('Pupil Phase')

    cax_pupil_phase = make_axes_locatable(ax_pupil_phase).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(SC, cax=cax_pupil_phase).ax.set_ylabel('Radians')

    # PSF Intensity
    ax_psf_intensity = plt.subplot(3, 4, 4*subplot-1)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    img_psf_intensity = plt.imshow(psf_intensity, norm=matplotlib.colors.LogNorm(vmin=5e-8, vmax=5e-5), aspect='equal')#, vmin=0, vmax=7e-10)
    plt.title('PSF Intensity')

    cax_psf_intensity = make_axes_locatable(ax_psf_intensity).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img_psf_intensity, cax=cax_psf_intensity).ax.set_ylabel('Energy (norm. units)')

    # Output modes
        # Scatter plot showing mock up of outputs
    ax_modes = plt.subplot(3, 4, 4*subplot, aspect='equal')
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    colour_map = 'plasma'
    # ax.set_facecolor(plt.get_cmap(colour_map)(0)) # Set it to the 0 of the colour map
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
    # cmap = plt.get_cmap(colour_map)
    # cmap.set_clim(vmin=0, vmax=0.35)
    vmax=0.35
    for x, y, c in zip(x_coords, y_coords, outputs):
        ax_modes.add_patch(matplotlib.patches.Circle((x, y), radius=0.5, facecolor=cm.plasma(c/vmax), edgecolor=None))
    # img_waveguides = ax.scatter(x_coords, y_coords, c=outputs, cmap=colour_map, vmin=0, vmax=0.35, edgecolors='r', linewidths=0.4)
    plt.title('Mock Output Waveguides')

    cax_modes = make_axes_locatable(ax_modes).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=vmax),
                cmap=colour_map), cax=cax_modes).ax.set_ylabel('Normalised Intensity')
    pass

def create_summaries(results, standard_descriptions, pretty_descriptions, outputs, filepath):
    # Save results
    result_arrays = [result.x for result in results]
    np.savez(filepath[:-3] + '{}_solutions.npz'.format(standard_descriptions[-1]), solutions=result_arrays, outputs=outputs)

    # Create debugging plots
    for i in range(0, len(results), 3):
        plt.figure(figsize=(20, 15))
        # plt.suptitle(model_name + ' Maximising Modes')
        plt.tight_layout()
        for j in range(min(3, len(results) - (i))):
            generate_summary(results[i+j],  standard_descriptions[i+j], j+1)
        plt.savefig(filepath[:-3] + '{}_{}.png'.format(standard_descriptions[-1], i))
        plt.close()

    # Create pretty plots
    for i in range(0, len(results), 3):
        plt.figure(figsize=(24, 15))
        for j in range(min(3, len(results) - (i))):
            generate_summary(results[i+j],  pretty_descriptions[i+j], j+1, pretty=True)
        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(hspace=0.1, wspace=0.3)
        plt.savefig(filepath[:-3] + '{}_{}_pretty.png'.format(standard_descriptions[-1], i))
        plt.close()

def weighted_mse_loss():
    print('THIS SHOULD NEVER BE RUN')
    pass

def explore_model(zernikes):
    filepath, model_name = load_model(zernikes)

    # Funnels
    # np.amax()
    amax_result = run_gd(zernikes, 'amax', nit=1000)
    # # L2
    # l2_result = run_gd(zernikes, 'L2')
    # # Funnelx3
    # funnel_x3_result = run_gd(zernikes, 'funnelx', 3)
    #
    # # Nulling
    # # Central Dim
    # nulling_central_result = run_gd(zernikes, 'nulling_central')
    # # np.amin()
    # amin_result = run_gd(zernikes, 'amin')
    # # Nullingx3
    # nulling_x3_result = run_gd(zernikes, 'nullingx', 3)

    # ***** CURRENTLY ONLY DOES AMAX
    results = [amax_result, ]
    # results = [run_gd(zernikes, 'funnelx', 7, nit=1000), ]

    spacing = ' '*20
    pretty_text = 'Maximising light into \nwaveguide 0\n\n{}{:.0%} of light \n{}funnelled'
    # pretty_text = 'Maximising light into \n7 waveguides\n\n{}{:.0%} of light \n{}funnelled'
    standard_text = 'waveguide 0'
    # standard_text = '7 waveguides'
    outputs = [get_fluxes(result.x)[0] for result in results]

    pretty_descriptions = [pretty_text.format(spacing[:14], np.amax(outputs[i]), spacing[:20]) for i in range(len(results))]
    # pretty_descriptions = [pretty_text.format(spacing[:14], np.sum(sorted(outputs[i], reverse=True)[:7]), spacing[:20]) for i in range(len(results))]
    standard_descriptions = [standard_text for i in range(len(results))]
    # standard_descriptions += ['funnel_7_limited']
    standard_descriptions += ['funnel_0_limited']

    create_summaries(results, standard_descriptions, pretty_descriptions, outputs, filepath)

def explore_modes(zernikes):
    filepath, model_name = load_model(zernikes)
    # Each run will have 3 summary panes: a text summary, the input wavefront and the output modes
    results = []
    modes = 19
    for i in range(modes):
        results += [run_gd(zernikes, 'maximise', i)]

    spacing = ' '*20
    pretty_text = 'Maximising light into \nwaveguide {}\n\n{}{:.0%} of light \n{}funnelled'
    standard_text = 'waveguide {}'
    outputs = [get_fluxes(result.x) for result in results]

    pretty_descriptions = [pretty_text.format(i, spacing[:14], np.amax(outputs[i]), spacing[:20]) for i in range(len(results))]
    standard_descriptions = [standard_text.format(i) for i in range(modes)]
    standard_descriptions += ['funnel']

    create_summaries(results, standard_descriptions, pretty_descriptions, outputs, filepath)

def explore_nulling(zernikes):
    filepath, model_name = load_model(zernikes)
    # Each run will have 3 summary panes: a text summary, the input wavefront and the output modes
    results = []
    null_modes = 19
    for i in range(1, null_modes):
        results += [run_gd(zernikes, 'nullingx', i)]

    spacing = ' '*20
    pretty_text = 'Darkening {} \nwaveguides\n\n{}{:.1%} of light\n{} let through'
    standard_text = '{} Modes'
    outputs = [get_fluxes(result.x)[0] for result in results]

    pretty_descriptions = [pretty_text.format(i, spacing[:14], np.sum(sorted(outputs[i], reverse=False)[:i+1]), spacing[:18]) for i in range(1, len(results) + 1)]
    standard_descriptions = [standard_text.format(i) for i in range(modes)]
    standard_descriptions += ['null']

    create_summaries(results, standard_descriptions, pretty_descriptions, outputs, filepath)

def find_specified(zernikes):
    filepath, model_name = load_model(zernikes)
    goal = (np.array([0.00943076, 0.01871392, 0.00712971, 0.01139204, 0.0155844,  0.01948092,
     0.00767647, 0.01675012, 0.01981159, 0.01752164, 0.01704332, 0.02802718, 0.02370392,
     0.02062199, 0.02510853, 0.00676503, 0.01889175, 0.01244253, 0.02157937])/0.46242796074329595)/2.138387
    result = run_gd(zernikes, 'specified', goal)
    plt.figure(figsize=(20, 15))
    plt.tight_layout()
    generate_summary(result, 'Specified', 1)
    plt.savefig(filepath[:-3] + '_specified.png')
    plt.close()

def rotate_zernikes():
    zernikes = 8
    filepath, model_name = load_model(zernikes)
    coeffs = np.array([-0.0359, 0.1989, -0.2665, -0.04, -0.5913, -0.1226, -0.0086])
    # coeffs = np.array([0, 0, 0, 0, 0, 0, 0])
    # angle = 60
    results = []
    angles = [0, 60, 120, 180, 240, 300]
    for angle in angles:
        # rotation_matrix = np.zeros((zernikes-1, zernikes-1))
        # rotation_matrix[0,0] = cos(1*radians(angle))
        # rotation_matrix[0,1] = sin(1*radians(angle))
        # rotation_matrix[1,0] = sin(-1*radians(angle))
        # rotation_matrix[1,1] = cos(-1*radians(angle))
        # rotation_matrix[2,2] = 1
        # rotation_matrix[3,3] = cos(2*radians(angle))
        # rotation_matrix[3,4] = sin(2*radians(angle))
        # rotation_matrix[4,3] = sin(-2*radians(angle))
        # rotation_matrix[4,4] = cos(-2*radians(angle))
        # rotation_matrix[5,5] = cos(1*radians(angle))
        # rotation_matrix[5,6] = sin(1*radians(angle))
        # rotation_matrix[6,5] = sin(-1*radians(angle))
        # rotation_matrix[6,6] = cos(-1*radians(angle))
        # print(np.matmul(rotation_matrix, coeffs.T))
        rotated_coeffs = np.zeros(zernikes-1)
        rotated_coeffs[0] = coeffs[0]*cos(radians(angle)) + coeffs[1]*sin(radians(angle))
        rotated_coeffs[1] = coeffs[1]*cos(-radians(angle)) + coeffs[0]*sin(-radians(angle))
        rotated_coeffs[2] = coeffs[2]
        # rotated_coeffs[3] = coeffs[3]*cos(2*radians(angle)) + coeffs[4]*sin(2*radians(angle))
        # rotated_coeffs[4] = coeffs[4]*cos(-2*radians(angle)) + coeffs[3]*sin(-2*radians(angle))
        rotated_coeffs[3] = coeffs[3]*cos(-2*radians(angle)) + coeffs[4]*sin(-2*radians(angle))
        rotated_coeffs[4] = coeffs[4]*cos(2*radians(angle)) + coeffs[3]*sin(2*radians(angle))
        # rotated_coeffs[5] = coeffs[5]*cos(radians(angle)) + coeffs[6]*sin(radians(angle))
        # rotated_coeffs[6] = coeffs[6]*cos(-radians(angle)) + coeffs[5]*sin(-radians(angle))
        rotated_coeffs[5] = coeffs[5]*cos(-radians(angle)) + coeffs[6]*sin(-radians(angle))
        rotated_coeffs[6] = coeffs[6]*cos(radians(angle)) + coeffs[5]*sin(radians(angle))
        print(rotated_coeffs)
        # coeffs = np.array([[-0.0359], [0.1989], [-0.2665], [-0.04], [-0.5913], [-0.1226], [-0.0086]])
        # print(np.matmul(rotation_matrix, coeffs))
        # input('')

        results += [SomeResult(rotated_coeffs, 0)]
    for i in range(0, len(results), 3):
        plt.figure(1, figsize=(20, 15))
        plt.tight_layout()
        plt.suptitle(model_name + ' Rotating ')
        generate_summary(results[i], 'Rotating {} Degrees'.format(angles[i]), 1)
        generate_summary(results[i+1], 'Rotating {} Degrees'.format(angles[i+1]), 2)
        generate_summary(results[i+2], 'Rotating {} Degrees'.format(angles[i+2]), 3)
        plt.savefig(filepath[:-3] + '_Rotated{}.png'.format(angles[i]))
        plt.close()

def compare_inputs():
    filepath, model_name = load_model(8)
    font = '26'
    plt.rcParams['font.size'] = font
    plt.rcParams['axes.titlesize'] = font
    plt.rcParams['axes.labelsize'] = font
    plt.rcParams['figure.titlesize'] = font

    plane_wave = [0, 0, 0, 0, 0, 0, 0]
    # plane_wave = [0, 0, 0, 0, 0, 0, 0, 0]
    # central_focus = [0.0112, 0.0111, -0.6319, -0.004, 0.0082, -0.005, 0.019]
    central_focus = [0.004, 0.014, -0.555, -0.007, 0.008, -0.006, -0.019]
    # central_focus_crafted = [0, 0, -0.6319, 0, 0, 0, 0]
    central_focus_crafted = [0, 0, -0.555, 0, 0, 0, 0]
    hesta_10_central = [-0.15, -0.38228785, -0.4, -0.15, -0.15, -0.33776387, -0.4]

    spacing = ' '*20
    # standard_text = '1 waveguides'
    # plane_results = [SomeResult(plane_wave, 'Plane_Wave'), SomeResult(central_focus, 'Found Solution')]
    # outputs = [get_fluxes(result.x)[0] for result in plane_results]
    # pretty_descriptions = [f'Plane wave\n\n{spacing[:14]}{outputs[0][0]:.0%} of light \n{spacing[:18]}in central \n{spacing[:18]}waveguide',
    #             f'Found solution\n\n{spacing[:14]}{outputs[1][0]:.0%} of light\n{spacing[:18]}in central \n{spacing[:18]}waveguide']
    #
    # standard_descriptions = ['Plane wave', 'Found solution']
    # standard_descriptions += ['_plane_waves']
    #
    # create_summaries(plane_results, standard_descriptions, pretty_descriptions, outputs, filepath)
    # input('Done plane wave')

    nulling_arrays = [[ 0.77259023, -0.04108601, -0.0080862 ,  0.03728083,  0.02458926,
       -0.00896498, -0.15386352],
       [ 0.7038555 , -0.08019237,  0.01180666, -0.00810761,  0.1079943 ,
        0.01332049, -0.10010728],
        [ 0.67387072, -0.01557461,  0.01006687, -0.00560442,  0.12258893,
        0.00288499, -0.09486048],
        [ 0.68681269, -0.0136801 ,  0.01936984, -0.01894881,  0.10127277,
       -0.00470996, -0.09377661],
       [ 0.68847746, -0.06586085,  0.02612743, -0.02890906,  0.06883536,
       -0.00163603, -0.10199869],
       [ 0.40728332, -0.34953641, -0.01042952, -0.17152243,  0.19612792,
        0.01779978, -0.05261913],
        [-0.11752445, -0.56642979,  0.01641481, -0.13954839, -0.15148539,
        0.05350868,  0.00796366],
        [ 0.38976102, -0.29988122, -0.03301733, -0.20123046,  0.2165151 ,
        0.02359298, -0.05006434],
        [ 0.37551952, -0.28830252, -0.04030202, -0.25295289,  0.17251215,
        0.01432587, -0.06113072],
        [ 0.37806518, -0.25201223, -0.02929857, -0.26643791,  0.18012218,
        0.00623554, -0.06180265],
        [ 0.35645908, -0.26303953, -0.06360745, -0.25537792,  0.23202244,
       -0.0250286 , -0.07741366],
       [ 0.35122256, -0.27755255, -0.07093549, -0.24003651,  0.2553212 ,
       -0.0116696 , -0.07918085],
       [ 0.34617855, -0.25795603, -0.08971017, -0.26281658,  0.23016305,
       -0.02587182, -0.09686488],
       [ 0.26292948, -0.16304072, -0.09135876, -0.37146477,  0.18234705,
        0.07107865, -0.11012987],
        [ 0.22301139, -0.15120836, -0.10064746, -0.46822249,  0.22718836,
        0.07786029, -0.13196906],
        [ 0.22352494, -0.17396711, -0.1281339 , -0.54940514,  0.13523805,
        0.08980382, -0.10701667],
        [ 0.22122636, -0.15169753, -0.14881817, -0.52330351,  0.28527576,
        0.06360916, -0.14454064],
        [ 0.26822891, -0.21866851, -0.07024552, -0.53488448,  0.24478153,
        0.08470067, -0.14464253]]

    funnel_arrays = [[-0.0019, 0.00134, -0.555, -0.0067, 0.0084, -0.0061, -0.0199],
            [0.3456, 0.1511, -0.0652, 0.4757, 0.3359, -0.0539, -0.1463],
            [0.2673, -0.219, -0.0714, -0.5365, 0.2434, 0.084, -0.1444],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [-0.2355, 0.0056, -0.2436, 0.0166, 0.6039, 0.0031, 0.1306],
            [-0.1924, 0.1376, -0.3049, -0.5391, 0.3289, -0.0407, 0.0924],
            [-0.1073, 0.2211, -0.2303, -0.506, -0.3, -0.1146, 0.0585]]

    filepath, model_name = load_model(11)
    apollo5_funnel = [0, 0, -0.5854, 0, 0, 0, 0, 0, 0, 0.2396]

    spacing = ' '*20
    pretty_text = 'Maximising light into \na single waveguide\n\n{}{:.0%} of light \n{}funnelled'
    standard_text = '1 waveguides'
    apollo_results = [SomeResult(apollo5_funnel, 'Funnel'), ]
    outputs = [get_fluxes(result.x)[0] for result in apollo_results]

    pretty_descriptions = [pretty_text.format(spacing[:14], np.amax(outputs[i]), spacing[:20]) for i in range(len(apollo_results))]
    standard_descriptions = [standard_text for i in range(1)]
    standard_descriptions += ['_7_funnel']
    # standard_descriptions += ['_plane_wave']

    create_summaries(apollo_results, standard_descriptions, pretty_descriptions, outputs, filepath)
    input('Done!')

    # # Plot nulling results
    # funelling_results = [SomeResult(arr, 'Funnels') for arr in funnel_arrays]
    # #
    # spacing = ' '*20
    # pretty_text = 'Maximising light into \nwaveguide {}\n\n{}{:.0%} of light\n{} funnelled'
    # standard_text = '{} Modes'
    # outputs = [get_fluxes(result.x)[0] for result in funelling_results]
    # #
    # pretty_descriptions = [pretty_text.format(i, spacing[:14], np.amax(outputs[i]), spacing[:18]) for i in range(len(funelling_results))]
    # standard_descriptions = [standard_text.format(i) for i in range(19)]
    # standard_descriptions += ['funnel']
    # #
    # create_summaries(funelling_results, standard_descriptions, pretty_descriptions, outputs, filepath)
    # input('Funnels done!')

    # # Plot nulling results
    # nulling_results = [SomeResult(arr, 'Nulling') for arr in nulling_arrays[:-1]]
    # #
    # spacing = ' '*20
    # pretty_text = 'Darkening {} \nwaveguides\n\n{}{:.0%} of light\n{} let through'
    # standard_text = '{} Modes'
    # outputs = [get_fluxes(result.x)[0] for result in nulling_results]
    # #
    # pretty_descriptions = [pretty_text.format(i+1, spacing[:14], np.sum(sorted(outputs[i], reverse=False)[:i+1]), spacing[:18]) for i in range(len(nulling_results))]
    # standard_descriptions = [standard_text.format(i) for i in range(19)]
    # standard_descriptions += ['null']
    # #
    # create_summaries(nulling_results, standard_descriptions, pretty_descriptions, outputs, filepath)
    # input('Nulls done!')

    # Plot Hyperion funnel results
    # filepath, model_name = load_model(9)
    # hyperion_solution = [[-0.8354622 ,  0.97509289, -0.93421231, -0.08981731, -0.20246804,
    #    -1.        , -0.11244781,  0.14261301]]
    hyperion_funnel_7 = [[-0.83164805,  1.        , -0.89603332, -0.0925162 , -0.20070668,
       -1.        , -0.11298515,  0.14612676]]
    # hyperion_results = [SomeResult(hyperion_solution[0], 'Central Funnel'), ]
    hyperion_results = [SomeResult(hyperion_funnel_7[0], '7 Funnel'), ]
    hyperion_plane_wave = [SomeResult(plane_wave, 'Plane Wave'), ]

    # spacing = ' '*20
    # pretty_text = 'Maximising light into \nwaveguide {}\n\n{}{:.0%} of light \n{}funnelled'
    # standard_text = 'waveguide {}'
    # outputs = [get_fluxes(result.x)[0] for result in hyperion_results]
    #
    # pretty_descriptions = [pretty_text.format(i, spacing[:14], np.amax(outputs[i]), spacing[:20]) for i in range(len(hyperion_results))]
    # standard_descriptions = [standard_text.format(i) for i in range(1)]
    # standard_descriptions += ['funnel']

    spacing = ' '*20
    pretty_text = 'Maximising light into \n7 waveguides\n\n{}{:.0%} of light \n{}funnelled'
    standard_text = '7 waveguides'
    outputs = [get_fluxes(result.x)[0] for result in hyperion_results]

    pretty_descriptions = [pretty_text.format(spacing[:14], np.sum(sorted(outputs[i], reverse=True)[:7]), spacing[:20]) for i in range(len(hyperion_results))]
    standard_descriptions = [standard_text for i in range(1)]
    standard_descriptions += ['_7_funnel']
    # standard_descriptions += ['_plane_wave']

    # create_summaries(hyperion_results, standard_descriptions, pretty_descriptions, outputs, filepath)

    # figure = plt.figure(figsize=(20, 15))
    # plt.suptitle(model_name + ' Comparison of Solution to Plane Wave')
    # plt.tight_layout()
    # # generate_summary(SomeResult(plane_wave, get_fluxes(plane_wave)[0][0]), 'Plane Wave', 1)
    # spacing = ' '*20
    # generate_summary(SomeResult(plane_wave, get_fluxes(plane_wave)[0][0]), f'Darkening 6 \nwaveguides\n\n{spacing[:14]}0.8% of light\n{spacing[:18]} let through', 1, pretty=True)
    # generate_summary(SomeResult(central_focus, get_fluxes(central_focus)[0][0]), 'Found Solution', 2)
    # generate_summary(SomeResult(central_focus_crafted, get_fluxes(central_focus_crafted)[0][0]), 'Crafted Solution', 3)
    # plt.savefig(filepath[:-3] + '_pretty_test.png')
    # # plt.show()
    # plt.close()

if socket.gethostname() == 'tauron':
    set_gpu(0)

# main(8)
# explore_model(8)
# explore_modes(8)
# explore_nulling(8)
# find_specified(8)
# rotate_zernikes()
compare_inputs()
