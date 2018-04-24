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

def SNR_Calculator(wavelength, spatial, err):
    """Constructs ndarray of SNRs such that SNR(redshift, linewidth)
    Parameters: wavelength: array of wavelength range over which to test the filter
                spatial: array of spetial dimension corresponding to the wavelength range
                err: 2D error array
                
    Returns: SNR: Signal-to-noise ratios of Amplitude
             z: Redshift array
			 width: linewidth array
    """
    
    lambda0_emitted = 3727.092 + (3729.875-3727.092)/2 #Midpoint of OII doublet
    
    #Initialise numpy arrays
    width = np.arange(0.1, 2.1, .1) #To calculate SNR at different linewidth
    z = np.zeros(len(wavelength))
    SNR = np.zeros((len(width), len(wavelength))) #linewidth vs z grid
    dataPrime = spatial/err #signal of data
    
    #Calculate SNR at different lambda0 and w
    for i in range(len(wavelength)):
        for j in range(len(width)):
            """Potentially solve the loop issue by starting with an 
			arbitary linewidth w and finding zmax; Then for given zmax,
			test different w"""
			
			lambda0 = wavelength[i]
            modelSpatial = Model((wavelength, lambda0), 1, width[j])
            modelPrime = modelSpatial/err
            
            """A = (SpatialPrime (dot) modelPrime)/(modelPrime (dot) modelPrime)
            sigmaA = 1/sqrt(modelPrime (dot) modelPrime)
            SNR = A/sigmaA"""
            sigmaA = 1./np.sqrt(np.dot(modelPrime, modelPrime))
            A = np.dot(dataPrime, modelPrime)/(sigmaA**(-2))
            SNR[j][i] = A/sigmaA
            
        #Convert lambda0 to z
        z[i] = lambda0/lambda0_emitted - 1
    
    return SNR, z, width
	
def DataProcessor(imagefile, errorfile, idx = 48):
	"""Uses Jae's code to process BinoSpec data
	Parameters: imagefile: Location of the 2D spectrum file
				errorfile: Location of the 2D error file
				idx: array number to be checked
	"""
	from utils import bino_data_preprocess
	
	data, header = bino_data_preprocess(imagefile, errorfile)
	image = data[idx][:, :, 0]
	err = data[idx][:, :, 1]
	ivar = 1/err**2 #2D inverse variance