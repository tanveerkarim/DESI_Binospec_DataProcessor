"""
This script generates:
1. SNR 2d plots as a function of w and z per slit per mask
2. txt files of redshifts and widths per mask
3. histograms of redshifts and widths per mask
"""

from utils_spec1d import datareader, SNR_calculator, plotterSNR2D, SNRvz, wave_grid
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sys
maskname = sys.argv[1]

#Check if directory for the mask exists; if not, create it
import os
if not os.path.exists('results/SNR2D/' + maskname):
	print('Mask directory does not exist. Creating directory...')
	os.makedirs('results/SNR2D/' + maskname)
	os.makedirs("results/SNRvsRedshift/" + maskname)
	os.makedirs("results/PeakZoom/" + maskname)
	
data = datareader(maskname)
image = data['data_ivar'][:, 0, :]
ivar = data['data_ivar'][:, 1, :]
datarows = len(image)

#Calculate the signal-to-noise
z_range, widths, SNRdata, Ampdata = SNR_calculator(maskname, data)

#Initalise arrays to store redshift and width values
zmax = np.zeros(datarows)
wmax = np.zeros(datarows)
		
#Time the code
from time import time
start = time()

wg = wave_grid(data) #wavelength grid of the 1d spectra

for i in range(1, datarows):
	plotterSNR2D(maskname, idx = i, z = z_range, widths = widths, SNRdata = SNRdata)
	zmax[i-1], wmax[i-1] = SNRvz(maskname, idx = i, z=z_range, widths=widths\
	, SNRdata=SNRdata, Ampdata=Ampdata, image=image, ivar=ivar, wavelength_grid=wg)
end = time()

tot_time = end - start
print(str(tot_time))

#Save redshift and width values in a txt file
import pandas as pd

df = pd.DataFrame({'max_z':zmax, 'max_w':wmax})
df.to_csv('results/Max_z_n_width/' + maskname + ".txt")

plt.hist(zmax[~np.isnan(zmax)], bins = 15, facecolor = 'red', alpha = 0.75)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Redshift histogram of ' + maskname)
plt.savefig('results/histograms/redshift/' + maskname + '.pdf', dpi = 600, bbox_inches = None)
plt.close()

plt.hist(wmax[~np.isnan(wmax)], bins = 4, facecolor = 'red', alpha = 0.75)
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.title('Width histogram of ' + maskname)
plt.savefig('results/histograms/width/' + maskname + '.pdf', dpi = 600, bbox_inches = None)
plt.close()