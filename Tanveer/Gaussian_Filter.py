import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
				idx: array number to be checked"""
	from utils import bino_data_preprocess
	
	data, header = bino_data_preprocess(imagefile, errorfile)
	image = data[idx][:, :, 0]
	err = data[idx][:, :, 1]
	ivar = 1/err**2 #2D inverse variance
	
	#Generate a plot of the 2D spectrum and inverse variance
	plt.clf()
	plt.subplot(2, 1, 1)
	plt.imshow(image, aspect = 'auto',  vmin=-4, vmax=4, cmap = 'gray', interpolation = 'None')
	plt.colorbar()
	plt.subplot(2, 1, 2)
	plt.imshow(ivar,  vmin=-4, vmax=4, aspect = 'auto', cmap = 'gray', interpolation = 'None')
	plt.colorbar()
	plt.savefig("Image+Ivar_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)

	#Generate physical wavelength grid
	Nobjs = 64
	i = 1
	crval1_600 = float(str(header).split("CRVAL1")[1].split("=")[1].split("/")[0])
	cdelt1_600 = float(str(header).split("CRVAL1")[1].split("=")[2].split("/")[0])
	
	for i in np.arange(2, Nobjs):
		tmp1 = float(str(header).split("CRVAL1")[1].split("=")[1].split("/")[0])
		tmp2 = float(str(header).split("CRVAL1")[1].split("=")[2].split("/")[0])    
		if np.abs(tmp1-crval1_600)>1e-6:
			print(tmp1)
		if np.abs(tmp2-cdelt1_600)>1e-6:
			print(tmp2)
			
	wave_grid_600 = crval1_600 + cdelt1_600 * np.arange(image[1].shape[0])
	wave_grid_600 *= 10
	
	#Generate profile and raw spectrum
	profile = image.sum(axis=1)
	plt.clf()
	plt.plot(profile)
	plt.savefig("Profile_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)
	
	spectrumRaw = image.sum(axis=0)
	plt.clf()
	plt.plot(wave_grid_600, spectrumRaw)
	plt.savefig("SpectrumRaw_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None
	
	
	#Generate cleaned spectrum
	spectrumClean = np.sum(np.multiply(image.T, np.multiply(ivar.T, profile)).T, axis = 0) \
	/np.sum(np.multiply(ivar.T, profile).shape)
	"""Need to figure out how to construct the 1D error array"""
	#errClean = 1/np.sqrt(np.sum(np.multiply(ivar.T, profile).T, axis = 0)) 
	errClean = 0.1 #For now use this dummy value
	plt.clf()
	plt.plot(wave_grid_600, spectrumClean)
	plt.savefig("SpectrumClean_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)
	
	return wave_grid_600, spectrumClean, errClean

idx = 48
wavegrid1d, spectra1d, err1d = DataProcessor("../../data/v1/st82-1hr-600/obj_abs_slits_lin.fits", 
"../../data/v1/st82-1hr-600/obj_abs_err_slits_lin.fits", idx)

start_time = time.time()
SNR, z, w = SNR_Calculator(wavegrid1d, spectra1d, err1d)
print("--- %s seconds ---" % (time.time() - start_time)) #Time taken to run code

#Generate grid plots
ax = sns.heatmap(SNR, linewidth=0.01)#, xticklabels=z, yticklabels=w)
ax.set(xlabel='Redshift z', ylabel='Line width w')
plt.clf()
plt.savefig("w_vs_z_grid1_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)
plt.clf()
plt.imshow(SNR, aspect='auto')
plt.colorbar()
plt.savefig("w_vs_z_grid2_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)

#Max SNR
print('Maximum SNR is ' + str(SNR.max()))

#Plot 1D spectra vs z for SNR.max
plt.plot(z, SNR[np.where(SNR == SNR.max())[0][0]])
plt.axhline(7)
plt.xlabel('z')
plt.ylabel('SNR')
plt.savefig("MaxSNR_" + str(idx) + ".pdf", dpi = 600, bbox_inches = None)