import h5py as h5
import numpy as np
import glob, os

from ecephys_spike_sorting.common.utils import printProgressBar

def convert_h5_to_dat(input_file, output_file, blocksize=100000):

	print('Reading from ' + input_file)

	data = h5.File(input_file)
	sample_count = data['data'].shape[0]
	num_blocks = np.ceil(sample_count / blocksize).astype('int')

	print('Writing ' + str(num_blocks) + ' blocks to ' + output_file)

	f = open(output_file, 'wb')

	for block in range(num_blocks):

		printProgressBar(block+1, num_blocks)

		start_index = blocksize * block
		end_index = start_index + blocksize

		d = data['data'][start_index:end_index,:,:]
		d2 = np.reshape(d, (d.shape[0], 384)).astype('int16')

		d2.tofile(f)

	f.close()

	print('Done')

input_files = glob.glob(r'H:\processed_3training\*.h5')

print(input_files)

for input_file in input_files:

	output_file = os.path.join(r'H:\processed_3training',
							'continuous.dat')

	output_directory = os.path.dirname(output_file)

	if not os.path.exists(output_directory):
		os.mkdir(output_directory)

	convert_h5_to_dat(input_file, output_file)
