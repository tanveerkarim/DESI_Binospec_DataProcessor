import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ModelO2(wavelength_array, width, Amp = 1):
	"""Returns the [OII] doublet model
	Parameters: wavelength_array: Full wavelength array of MMT BinoSpec. This is constant.
				width: width of the Gaussian doublets. This varies.
				Amp: Amplitude of the Gaussian doublets
	"""

	Gaussian = lambda x, mean, std: np.exp(-((x - mean)/std)**2)

	#Values from http://classic.sdss.org/dr6/algorithms/linestable.html
	separation = (3729.875-3727.092)/2 #separation between lambda0 and the emission lines
	lambda0 = wavelength_array[(len(wavelength_array)-1)//2] #Take the midpoint of the wavelength_array as the centre
                                                             #of the Gaussian doublet
	return Amp*(Gaussian(wavelength_array, lambda0-separation, width) + Gaussian(wavelength_array, lambda0+separation, width))

def Window(wavelength_array, ngal, pixel_size, window_size_multiplier):
	"""Returns windows to run the Model function over to speed up calculation
	Parameters: wavelength_array: Full wavelength array of MMT BinoSpec. This is constant
				ngal: number of galaxies in a given data file
				pixel_size: width of pixels in wavelength_array
				window_size_multiplier: Multiple this with pixel size to get width of window in wavelength_array space
	Returns: nwindow_ndarray: l x m x n ndarray where l = ngal, m = number of windows and 
							n = (pixel_size*window_size_multiplier)//2 + 1
	"""
	
	nwindow = len(wavelength_array) - window_size_multiplier//2 #number of windows per galaxy. 
													#It is of this form b/c beyond this window exceeds the wavelength_array
	nwindow_array = np.zeros((nwindow, (pixel_size*window_size_multiplier)//2 + 1))
	
	#Generate nwindow windows of size ((pixel_size*window_size_multiplier)//2 + 1)
	for i in range(nwindow):
		nwindow_array[i] = np.arange(wavelength_array[i], wavelength_array[i] + pixel_size*(1 + window_size_multiplier), \
									 pixel_size)
	
	print(nwindow_array[0])
	#Repeat nwindow_arary ngal times 
	#https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
	nwindow_ndarray = np.repeat(nwindow_array[np.newaxis, :, :], ngal, axis=0)

	return nwindow_ndarray