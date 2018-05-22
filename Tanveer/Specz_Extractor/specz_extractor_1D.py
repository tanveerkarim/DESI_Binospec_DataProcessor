import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ModelO2(wavelength_array, width, Amp = 1):
"""Returns the [OII] doublet model
	Parameters: wavelength_array: Full wavelength array of MMT BinoSpec. This is constant.
				width: width of the Gaussian doublets. This varies.
				Amp: Amplitude of the Gaussian doublets"""

	Gaussian = lambda x, mean, std: np.exp(-((x - mean)/std)**2)
	
	#Values from http://classic.sdss.org/dr6/algorithms/linestable.html
	separation = (3729.875-3727.092)/2 #separation between lambda0 and the emission lines
	
	return Amp*(Gaussian(wavelength_array, lambda0-separation, std) + Gaussian(spectra, lambda0+separation, std))


