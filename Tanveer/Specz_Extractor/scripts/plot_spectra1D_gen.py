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
	import numpy as np
	import matplotlib.pyplot as plt
	
	def datareader(maskname):
		"""Reads mask data for use by the other functions in this module
		Parameters
		----------
		maskname: name of the mask + '-' + grating number
		"""
		
		fname = maskname + '.npz'
		data = np.load(fname)
		
		return data
		
	data_dir = "../../../DATA_MAY18/spec1d/"
	fname270 = '2-8h30m-270-spec1d.npz'
	fname600 = '2-8h30m-600-spec1d.npz'
	data270 = np.load(data_dir + fname270)
	data600 = np.load(data_dir + fname600)
	
	def wave_grid(data):
		"""Returns wavegrid based on header file from data"""
		
		crval1 = float(str(data['headers'][1]).split("CRVAL1")[1].split("=")[1].split("/")[0]) #Starting value
		cdelt1 = float(str(data['headers'][1]).split("CDELT1")[1].split("=")[1].split("/")[0]) #Pixel size
		
		collapsedSpectrum = data['data_ivar'][:, 0, :]
		
		wave_grid = crval1 + cdelt1 * np.arange(collapsedSpectrum[1].shape[0])
		wave_grid *= 10 #Convert wave_grid to Angstrom from nm
		return wave_grid
	
	wg270 = wave_grid(data270)
	wg600 = wave_grid(data600)
	
	#User defines which grating to generate
	wg_key = int(input("Enter grating number: "))
	
	if(wg_key == 270):
		wg = wg270
	elif(wg_key == 600):
		wg = wg600
	else:
		print("Enter 270 or 600.")
	
	def plotterSpectra1D(maskname, data, idx):
		"""Returns plots of the 1d spectra and the inverse variance
		
		Parameters
		----------
		idx: Index number of the slit; ignore idx = 0
		"""
		
		image = data['data_ivar'][:, 0, :]
		ivar = data['data_ivar'][:, 1, :]
		
		imagetmp = image[idx, :]
		ivartmp = ivar[idx, :]
		
		f, axarr = plt.subplots(2, sharex=True)
		axarr[0].plot(wg, imagetmp)
		axarr[0].set_title('Mask: ' + str(maskname) + ', ' + 'Slit ' + str(idx) + "\n" + "1D spectra" \
						,  fontsize = 15, fontname = 'serif')
		axarr[1].plot(wg, ivartmp)
		axarr[1].set_title('1D inverse variance', fontsize = 15, fontname = 'serif')
		plt.savefig('results/spectra1d/' + str(maskname) + '/' + str(maskname)\
		+ '-' + str(idx) + '-spectra1d.pdf', dpi = 600, bbox_inches = None)
		plt.close()
	
	data = datareader(maskname)
else:
	data = datareader(maskname)

datarows = len(data['data_ivar'][:, 0, :])

#Time the code
#from time import time
#start = time()

for i in range(1, datarows):
	plotterSpectra1D(maskname, data, idx = i)
	
#end = time()
#tot_time = end - start
#print(str(tot_time))