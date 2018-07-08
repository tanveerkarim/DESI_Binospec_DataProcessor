"""This script generates 1D spectra and inverse variance plots for a given mask
"""

from specz_extractor_1D import datareader, plotterSpectra1D
import sys
maskname = sys.argv[1]

#Check if directory for the mask exists; if not, create it
import os
if not os.path.exists('results/spectra1d/' + maskname):
	print('Mask directory does not exist. Creating directory...')
	os.makedirs('results/spectra1d/' + maskname)

isMock = True
if(isMock == True):
	
	def datareader(maskname):
		"""Reads mask data for use by the other functions in this module
		Parameters
		----------
		maskname: name of the mask + '-' + grating number
		"""
		
		fname = maskname + '.npz'
		data = np.load(fname)
		
		return data
	
	data = datareader(maskname)
else:
	data = datareader(maskname)

datarows = len(data['data_ivar'][:, 0, :])

#Time the code
from time import time
start = time()

for i in range(1, datarows):
	plotterSpectra1D(maskname, data, idx = i)
	
end = time()

tot_time = end - start
print(str(tot_time))