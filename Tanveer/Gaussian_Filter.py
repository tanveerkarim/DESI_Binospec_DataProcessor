import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

def ModelDoublet(params, Amp = 1, std = 0.75): #Default, A = 1, std = 0.75
    """Returns the [OII]
    Parameters: Amp: Amplitude of the Gaussian
				std: sigma of the Gaussian
                params: tuple of wavelength range and lambda0 (centre point of the doublet Model)
    """
    
    wavelength, lambda0 = params
    Gaussian = lambda x, mean, std: np.exp(-((x - mean)/std)**2)
    
    #Values from http://classic.sdss.org/dr6/algorithms/linestable.html
    separation = (3729.875-3727.092)/2 #separation between lambda0 and the emission lines
    
    return Amp*(Gaussian(wavelength, lambda0-separation, std) + Gaussian(wavelength, lambda0+separation, std))

