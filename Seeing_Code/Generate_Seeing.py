"""
An example using Opticspy to fit a set of Zernike terms to some arbitrary phase map.
This can be used for the photonic lantern projects for decomposing seeing into
Zernike terms for subseqent use in a Zernike-basis regression (e.g. NN), or to match
measured performance to theoretical best case scenario.

Opticspy can be downloaded from
https://github.com/Sterncat/opticspy
With docs at
http://opticspy.org

I found that opticspy functions which used mplot3d were flaky, probably a matplotlib
version incompatability. But you don't need opticspy's 3D plotting just for this
task.
"""

import opticspy
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
plt.ion()

def create_atmosphere():
	D_tel = 8.2 # meter
	pupil_grid = make_pupil_grid(512, D_tel)

	fried_parameter = 2 # meter
	outer_scale = 20 # meter
	velocity = 0.1 # meter/sec

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
	layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
	return layer

def get_seeing_at_time(wavelength, atmosphere, time):
	atmosphere.evolve_until(time)
	output_phase = atmosphere.phase_for(wavelength)
	return np.reshape(output_phase, (512, -1))

def decompose_over_zernikes(seeing):
	# Ok now fit Zernike polynomials
	# WARNING - be careful about which term is which! The first 'n' terms as enumerated here
	# are not necessarily the same as the first 'n' terms used in other code (e.g. poppy)

	num_terms = 8 # Number of Zernike terms to fit (does not include piston).
	removepiston = True # Remove the piston term?
	remain2D = False # Show a plot of the residuals (what is left over after the fit)?
	barchart = False # Show a bar chat of fitted polynomial terms?

	fit_list, _ = opticspy.zernike.fitting(seeing, num_terms, removepiston=removepiston,
		remain2D=remain2D, barchart=barchart)

	return fit_list

def convert_to_noll_order(zernike):
	noll_ordered_zernike = [zernike[1], zernike[2], zernike[3], zernike[5], zernike[4], zernike[7], zernike[6]]
	return noll_ordered_zernike


# Get some phase map to fit to. Here just use a spherical map as a test.
size = 256
wavelength = 1500e-9
# phase_map = make_pupil_grid(size)
# atmosphere = AtmosphericLayer(phase_map, Cn_squared=1)
# atmosphere = make_standard_atmospheric_layers(phase_map)
atmosphere_layer = create_atmosphere()


# plt.figure()
# plt.imshow(phase_map)
# plt.colorbar()
# plt.title('Input phase map')

seeings = []
zernikes = []
for half_minute in range(20):
	for time in np.arange(half_minute*30, 30 + half_minute*30, 1/30):
		seeing = get_seeing_at_time(wavelength, atmosphere_layer, time)
		# plt.figure(1)
		# plt.imshow(seeing, cmap='RdBu', vmin=-2, vmax=2)
		# plt.title('Seeing')
		# if time == 0:
		# 	plt.colorbar()
		# plt.show()
		# plt.pause(0.01)
		zernike = decompose_over_zernikes(seeing)
		zernike = convert_to_noll_order(zernike)
		seeings += [seeing]
		zernikes += [zernike]
		print('*************')
		print(f'Time = {time:.2f}')
		# print(zernike)
		# print('*************')

	np.savez(f'/media/tintagel/david/opticspy/seeing_10min_{half_minute}.npz', seeings=np.array(seeings), zernikes=np.array(zernikes))
	seeings = []
	zernikes = []
