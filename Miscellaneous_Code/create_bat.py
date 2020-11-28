from os import listdir
from math import ceil

def divide_list(ls, n):
	# ls is the list, n the number of chunks
	chunk_size = ceil(len(ls)/n)
	for i in range(0, len(ls), chunk_size):
		yield ls[i:i+chunk_size]

directory = 'Z:\\david\\BigOne2'
ind_file_name = '19CorePL_June2019_realhex_bigone2.ind'
prefix = 'ten_zernike_zernikePSFs_-0.0413_0.0155'
fld_files = listdir(directory)
num_of_bat_files = 31
fld_files = list(enumerate(list(divide_list(fld_files, num_of_bat_files))))
# fld_file_subset has form (0, [...])
for fld_file_subset in fld_files:
	with open(directory+'\\ten_zernike_{}.bat'.format(fld_file_subset[0]), 'w') as file:
		for fld_file in fld_file_subset[1]:
			prefix = 'ten_zernike_' + fld_file[:-15]
			file.write('bsimw32 {} prefix={} launch_file={} wait=0\n'.format(ind_file_name, prefix, fld_file))
with open(directory+'\\runAllBatfiles.bat', 'w') as file:
	for fld_file_subset in fld_files:
		file.write('start cmd /k call ten_zernike_{}.bat\n'.format(fld_file_subset[0]))
