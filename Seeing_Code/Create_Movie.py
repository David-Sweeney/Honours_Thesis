import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# import poppy
# import imageio
from math import sqrt, cos, sin, radians, pi
from socket import gethostname
from os import mkdir
# plt.ion()

def get_filepath():
    computer_name = gethostname()
    if computer_name == 'tauron':
        return '/media/tintagel/david'
    else:
        return '/import/tintagel3/snert/david'

def create_directory(out_path=f'{get_filepath()}/opticspy/frames'):
	# Create new directory in which to put image of each frame
	try:
		mkdir(out_path)
	except OSError as e:
		print(e)
		print()
		print('Unable to create new directory', out_path, 'because directory already exists.')
		input('')

def create_PL_output_plot(fluxes, i):
    fig = plt.figure(figsize=(10,10))
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    ax_modes = fig.add_axes([0, 0, 1, 1])
    # ax_modes.axis('off')

    # PL Outputs
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
    for x, y, c in zip(x_coords, y_coords, fluxes):
        ax_modes.add_patch(matplotlib.patches.Circle((x, y), radius=0.5, facecolor=cm.plasma(c/vmax), edgecolor=None))

    # cax_modes = make_axes_locatable(ax_modes).append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=vmax),
    #             cmap=colour_map), cax=cax_modes).ax.set_ylabel('Normalised Flux')

    # plt.savefig(f'{get_filepath()}/opticspy/10min_PL_frames/frame_{frame_number}.png')
    plt.savefig(f'{get_filepath()}/opticspy/outputs.png')
    plt.close()

def create_plot(frame_number, seeing, zernike, psf, flux_interval, time_interval):
    # Seeing
    # PSF
    # Flux line plot
    # PL Outputs
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.suptitle(f'Time = {time_interval[-1]:.1f}s')

    # Seeing
    ax_seeing = plt.subplot(221)
    img_seeing = plt.imshow(seeing, vmin=-1.5, vmax=1.5, cmap='RdBu', extent=[-4.1, 4.1, -4.1, 4.1])
    plt.xlabel('Telescope Pupil X Position (m)')
    plt.ylabel('Telescope Pupil Y Position (m)')
    plt.title('Seeing with windspeed 10m/s')

    cax_seeing = make_axes_locatable(ax_seeing).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img_seeing, cax=cax_seeing).ax.set_ylabel('Phase (radians)')

    # PSF
    ax_psf_intensity = plt.subplot(222)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    img_psf_intensity = plt.imshow(psf, norm=matplotlib.colors.LogNorm(vmin=5e-8, vmax=5e-5), aspect='equal')
    plt.title('PSF Intensity')

    cax_psf_intensity = make_axes_locatable(ax_psf_intensity).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img_psf_intensity, cax=cax_psf_intensity).ax.set_ylabel('Normalised Energy Units')

    # Flux line plot
    plt.subplot(223)
    selected_fluxes = np.array([flux_interval[:, 0], flux_interval[:, 1],
                                flux_interval[:, 8], flux_interval[:, 9]]).T
    plt.plot(time_interval, selected_fluxes)
    plt.ylabel('Normalised Intensity')
    plt.xlabel('Time (s)')
    if time_interval[-1] < 5:
        plt.xlim([0, 5])
    else:
        plt.xlim([time_interval[0], time_interval[-1]])
    plt.ylim([0, 0.25])
    plt.title('Selected Intensity vs Time')
    plt.legend(['Waveguide 0', 'Waveguide 1', 'Waveguide 8', 'Waveguide 9'], loc='upper left')

    # PL Outputs
    ax_modes = plt.subplot(224, aspect='equal')
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
    for x, y, c in zip(x_coords, y_coords, flux_interval[-1]):
        ax_modes.add_patch(matplotlib.patches.Circle((x, y), radius=0.5, facecolor=cm.plasma(c/vmax), edgecolor=None))
    plt.title('Photonic Lantern Outputs')

    cax_modes = make_axes_locatable(ax_modes).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=vmax),
                cmap=colour_map), cax=cax_modes).ax.set_ylabel('Normalised Flux')

    plt.savefig(f'{get_filepath()}/opticspy/frames/frame_{frame_number}.png')
    print('Frame', frame_number, 'completed.')
    # fig.canvas.draw()
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return image

def create_film_strip(frame, seeing, zernike, psf, flux, fig):
    # Seeing
    # PSF
    # PL Outputs
    print('Strip', frame)
    # fig, ax = plt.subplots(figsize=(15, 5))

    # Seeing
    # ax_seeing = fig.add_subplot(10, 3, frame*3+1)
    ax_seeing = fig.add_subplot(5, 3, frame*3+1)
    img_seeing = plt.imshow(seeing, vmin=-1.5, vmax=1.5, cmap='RdBu', extent=[-4.1, 4.1, -4.1, 4.1])
    if frame == 4:
        plt.xlabel('Pupil X Position (m)')
    else:
        plt.xticks([])
    plt.ylabel('Pupil Y Position (m)')
    if frame == 0:
        plt.title('Seeing w/ windspeed 10m/s')

    cax_seeing = make_axes_locatable(ax_seeing).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img_seeing, cax=cax_seeing).ax.set_ylabel('Phase (radians)')

    # PSF
    ax_psf_intensity = fig.add_subplot(5, 3, frame*3+2)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
    left=False, right=False, labelbottom=False, labelleft=False)
    img_psf_intensity = plt.imshow(psf, norm=matplotlib.colors.LogNorm(vmin=5e-8, vmax=5e-5), aspect='equal')
    if frame == 0:
        plt.title('PSF Intensity')

    cax_psf_intensity = make_axes_locatable(ax_psf_intensity).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img_psf_intensity, cax=cax_psf_intensity).ax.set_ylabel('Energy (norm. units)')

    # PL Outputs
    ax_modes = fig.add_subplot(5, 3, frame*3+3, aspect='equal')
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

    vmax=0.105
    for x, y, c in zip(x_coords, y_coords, flux):
        ax_modes.add_patch(matplotlib.patches.Circle((x, y), radius=0.5, facecolor=cm.plasma(c/vmax), edgecolor=None))
    if frame == 0:
        plt.title('Photonic Lantern Outputs')

    cax_modes = make_axes_locatable(ax_modes).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0, vmax=vmax),
                cmap=colour_map), cax=cax_modes).ax.set_ylabel('Normalised Flux')

    # plt.savefig(f'{get_filepath()}/opticspy/film_strip.png')

    return

# seeings, zernikes, fluxes, psfs = [], [], [], []
# print('Starting!')
# frame_number = 0
# for file in [f'seeing_10min_{i}.npz' for i in range(20)]:
#     print('***************************************')
#     print(f'Starting on {file}')
#     npz_file = np.load(f'{get_filepath()}/opticspy/{file}')
#     # seeings = npz_file['seeings']
#     # zernikes = npz_file['zernikes']
#     fluxes = npz_file['fluxes']
#     # psfs = npz_file['psfs']
#     for flux in fluxes:
#         create_PL_output_plot(flux, frame_number)
#         if frame_number % 100 == 0:
#             print('Frame', frame_number, 'completed.')
#         frame_number += 1
# print('Completed!')

create_PL_output_plot([0]*19, 0)

# print('Loading in file')
# npz_file = np.load(f'{get_filepath()}/opticspy/seeing_30s_fixed.npz')
# seeings = npz_file['seeings']
# zernikes = npz_file['zernikes']
# fluxes = npz_file['fluxes']
# psfs = npz_file['psfs']
# null_plot(fluxes)
# input('Completed null plot!')
# print('Starting film strip')
#
# font = '18'
# plt.rcParams['font.size'] = font
# plt.rcParams['axes.titlesize'] = font
# plt.rcParams['axes.labelsize'] = font
#
# rows = 5
# fig, big_axes = plt.subplots( figsize=(19.7, rows*5) , nrows=rows, ncols=1, sharey=True)
#
# for row, big_ax in enumerate(big_axes, start=1):
#     big_ax.set_ylabel(f"Time={row*7/30:.1f}\n", fontsize=26, labelpad=30)
#
#     # Turn off axis lines and ticks of the big subplot
#     # obs alpha is 0 in RGBA string!
#     # big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
#     big_ax.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,
#     left=False,
#     right=False,         # ticks along the top edge are off
#     labelbottom=False,
#     labelleft=False) # labels along the bottom edge are off
#     # removes the white frame
#     big_ax._frameon = False
# plt.tight_layout(pad=0, h_pad=0, w_pad=0)
# plt.subplots_adjust(hspace=0, wspace=0)
# for i in range(rows):
#     create_film_strip(i, seeings[i*7], zernikes[i*7], psfs[i*7], fluxes[i*7], fig)
# plt.tight_layout(pad=2, h_pad=0, w_pad=0)
# plt.subplots_adjust(hspace=0, wspace=0.3)
# plt.savefig(f'{get_filepath()}/opticspy/film_strip_big.png')

# time = np.array(list(range(len(fluxes))))/30
# time_intervals = []
# flux_intervals = []
# for i in range(int(len(seeings))):
#     if time[i] < 5:
#         time_interval = time[:i+1]
#         flux_interval = fluxes[:i+1]
#     else:
#         time_interval = time[(i+1)-5*30:i+1]
#         flux_interval = fluxes[(i+1)-5*30:i+1]
#     time_intervals += [time_interval]
#     flux_intervals += [flux_interval]

# print('Starting!')
# create_directory()
# frames = [create_plot(n, *i) for n, i in enumerate(zip(seeings, zernikes, psfs, flux_intervals, time_intervals))]
# print('Frames created!')
# # imageio.mimsave(f'{get_filepath()}/opticspy/video.gif', frames, fps=30)
# print('Done!')
