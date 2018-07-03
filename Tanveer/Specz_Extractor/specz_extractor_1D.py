import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def datareader(maskname, dir_name = "../../../DATA_MAY18/spec1d/"):
	"""Reads mask data for use by the other functions in this module
	Parameters
	----------
	maskname: name of the mask + '-' + grating number
	"""
	
	fname = maskname + '-' + 'spec1d.npz'
	data = np.load(dir_name + fname)
	
	return data

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

def Window(z, wg, z_grid, window_width = 0.005):
    """Returns a range of pixel in the specified window width
    
    Parameters
    ----------
    z: Centre of the window
    wg: wave grid that needs to be windowed
    z_grid: redshift grid of the wave_grid
    window_width: size of the window in redshift space
    
    Returns
    -------
    windowed_array: windowed array of the windowing_array    
    """

    windowed_array = wg[(z_grid > (z - window_width)) & (z_grid < (z + window_width))]
    
    return windowed_array
	
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

def SNR_calculator(data):
    z_range = np.arange(0.7, 1.6, 0.0001)
    widths = np.arange(.5, 1., .1)
    
    #Read data
    image = data['data_ivar'][:, 0, :]
    ivar = data['data_ivar'][:, 1, :]
    wg = wave_grid(data)
    z_grid = lambda_to_z(wg) #Convert wavelength space to redshift space
    
    results = np.zeros((z_range.size, image.shape[0], widths.size))
    
    for i, z in enumerate(z_range):
        wg2 = Window(z, wg, z_grid)
        
        model = Model(z, wg2, widths)
        
        #Find the idx of the edges of the windows and slice the image file to multiply with modelPrime
        minidx = np.where(wg == np.min(wg2))[0][0] 
        maxidx = np.where(wg == np.max(wg2))[0][0]
        imageSliced = image[:,minidx:maxidx+1]
        imageSliced = imageSliced[:, :, np.newaxis] #Broadcasting
        ivarSliced = ivar[:,minidx:maxidx+1]
        ivarSliced = ivarSliced[:, :, np.newaxis] #Broadcasting
        imagePrimeSliced = imageSliced*np.sqrt(ivarSliced)
        
        Mprime = np.sqrt(ivarSliced)*model
        Denominator = Mprime**2
        Denominator = np.sum(Denominator, axis = 1)
        Numerator = Mprime*imagePrimeSliced
        Numerator = np.sum(Numerator, axis = 1)
        
        """
        sigmaA^(-2) = M'.M'
        A = (D'.M')/(M'.M') => (D'.M')*(sigmaA^(2))
        """
        
        Amp = Numerator/Denominator
        sigmaA = np.sqrt(1./Denominator)
        SNR = Amp/sigmaA
        
        results[i] = SNR
        
    results = results.reshape((image.shape[0], z_range.size, widths.size))
    
    return results

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
	axarr[0].plot(wave_grid(data), imagetmp)
	axarr[0].set_title('Mask: ' + maskname + ', ' + 'Slit ' + str(idx) + "\n" + "1D spectra" \
					,  fontsize = 15, fontname = 'serif')
	axarr[1].plot(wave_grid(data), ivartmp)
	axarr[1].set_title('1D inverse variance', fontsize = 15, fontname = 'serif')
	plt.savefig('results/spectra1d/' + maskname + '/' + maskname + '-' + str(idx) + '-spectra1d.pdf', dpi = 600, bbox_inches = None)
	plt.close()

def plotterSNR2D(maskname, SNRdata, idx):
	"""Returns redshift vs. width plot with SNR strength.
	
	Parameters
	----------
	SNRdata: SNR datacube to be plotted. Output of function SNR_calculator.
	idx: Galaxy index that is to be plotted from SNRdata. 
	
	Returns
	-------
	PDF of the 2D plot
	"""
	
	plt.imshow(SNRdata[idx], aspect='auto', interpolation='None', \
			extent=[np.min(z), np.max(z), np.min(widths), np.max(widths)], vmin=0, vmax=7)
	plt.colorbar()
	plt.ylabel('width', fontsize = 15, fontname = 'serif')
	plt.xlabel('redshift', fontsize = 15, fontname = 'serif')
	plt.title('Mask: ' + maskname + ', ' + 'Slit ' + str(idx),  fontsize = 15, fontname = 'serif')
	plt.savefig("results/SNR2D/"  + maskname + '/' + maskname + '-' + str(idx) + "-SNR2d.pdf", dpi = 600, bbox_inches = None)
	