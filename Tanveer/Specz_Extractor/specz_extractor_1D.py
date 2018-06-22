import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def wave_grid(data):
	"""Returns wavegrid based on header file from data"""
	
	crval1 = float(str(data['headers'][1]).split("CRVAL1")[1].split("=")[1].split("/")[0]) #Starting value
	cdelt1 = float(str(data['headers'][1]).split("CDELT1")[1].split("=")[1].split("/")[0]) #Pixel size
	
	collapsedSpectrum = data['data_ivar'][:, 0, :]
	
	wave_grid = crval1 + cdelt1 * np.arange(collapsedSpectrum[1].shape[0])
	wave_grid *= 10 #Convert wave_grid to Angstrom from nm
	
	return wave_grid

def lambda_to_z(wavelength):
	"""Converts wavelength grid to redshift grid"""
	
	separation = (3729.875-3727.092)/2 #separation between the emission lines
	lambda0 = 3727.092 + separation #Midpoint of the gaussian emission lines in restframe
	
	return (wavelength/lambda0 - 1)

def Model(z, wg2, width, Amp = 1):
    """Returns Gaussian filter model at redshift z
    
    Parameters
    ----------
    z: array of redshifts at which the model is being tested
    wg2: pixel grid of the Window
    width: width array of the Gaussian doublets
    Amp: amplitude of the Gaussian doublets
    
    Returns
    --------
    model: Gaussian models in the range of [z - window_width, z + window_width]
    """
    
    lambda_r27 = 3727.092; lambda_r29 = 3729.875 #rest frame wavelength of the [OII] doublets
    separation_r = (lambda_r29 - lambda_r27) #separation between the emission lines in rest frame
    lambda0 = lambda_r27 + separation_r/2 #Midpoint of the gaussian emission lines in restframe
    lambda_obs = lambda0*(1 + z) #Observed wavelength of of the midpoint
    Gaussian = lambda x, mean, std: (1/np.sqrt(2*np.pi*std**2))*np.exp(-((x[:, np.newaxis] - mean)/std)**2)

    model = Amp/2*(Gaussian(wg2, lambda_obs - separation_r, width) + Gaussian(wg2, lambda_obs + separation_r, width))
        
    return model

def Window(wavelength_array, ngal, pixel_size, window_size_multiplier):
		"""Returns windows to run the Model function over to speed up calculation
		Parameters: wavelength_array: Full wavelength array of MMT BinoSpec. This is constant
					ngal: number of galaxies in a given data file
					pixel_size: width of pixels in wavelength_array
					window_size_multiplier: Multiple this with pixel size to get width of window in wavelength_array space
		Returns: nwindow_ndarray: l x m x n ndarray where l = ngal, m = number of windows and 
								n = pixels per windowsize
		"""

		nwindow = (wavelength_array[-1] - wavelength_array[0])//pixel_size #number of windows per galaxy. 
														#It is of this form b/c beyond this window exceeds the wavelength_array
		nwindow_array = []

		#Generate nwindow windows
		for i in range(nwindow):
			tmp = np.arange(wavelength_array[i], wavelength_array[i] + pixel_size*(window_size_multiplier), \
										 pixel_size)
			if(tmp[-1] > (wavelength_array[-1] + pixel_size)):
				break
			else:
				nwindow_array.append(tmp)

		nwindow_array = np.asarray(nwindow_array)
		
		#Repeat nwindow_arary ngal times 
		#https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
		nwindow_ndarray = np.repeat(nwindow_array[np.newaxis, :, :], ngal, axis=0)

		return nwindow_ndarray
		

		
