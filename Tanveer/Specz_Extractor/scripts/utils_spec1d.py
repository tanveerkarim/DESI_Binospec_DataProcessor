"""This module contains all the necessary functions to do MMT BinoSpec 1D analysis.
Authors: Tanveer Karim and Jae Lee
Latest version: 09-Jul-2018"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage.filters import median_filter
import os
plt.style.use('ggplot')

"""------START OF TANVEER'S CODE------"""

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

def SNR_calculator(maskname, data):
	
	"""maskname[-3:] yields the grating number. z_range changes depending
	on maskname because of variation in grating. The start and end points
	are chosen by inspecting the header file of the data."""
	
	if(maskname[-3:] == '270'):
		z_range = np.arange(0.677, 1.5, 0.001)
	elif(maskname[-3:] == '600'):
		z_range = np.arange(0.7, 1.61, 0.001)
	
	"""Gaussian width, sigma = sqrt(sigma_lambda^2 + sigma_slit^2) where, 
	sigma_lambda = sigma_v/c*lambda(z); sigma_v = [0, 300] km/s
	sigma_slit = 3.3/sqrt(12)*delLambda_pixel	
	"""
	
	delLambda_pixel = float(str(data['headers'][1]).split("CDELT1")[1]\
		.split("=")[1].split("/")[0])*10. #size of the pixel in angstrom
	sigma_slit = 3.3/sqrt(12)*delLambda_pixel
	sigma_v = np.arange(0, 301, 50) #[0, 300] km/s in steps of 50 km/s
	c = 299792.458 #km/s
	#rest frame wavelength of the [OII] doublets
	lambda_r27 = 3727.092; 
	lambda_r29 = 3729.875 
	separation_r = (lambda_r29 - lambda_r27) #separation between the emission lines in rest frame
	lambda0 = lambda_r27 + separation_r/2 #Midpoint of the gaussian emission lines in restframe
			
	def widthlist(z):
		"""Returns an array of possible Gaussian widths for the [OII] doublet
		model testing"""
		
		def lambda_obs(z):
			"""Returns lambda observed of the Gaussian doublet centroid as a 
			function of redshift"""
			
			return lambda0*(1 + z)
		
		sigma_lambda = sigma_v/c*lambda_obs(z)
	
		return np.sqrt(sigma_lambda**2 + sigma_slit**2)
	
	#Read data
	image = data['data_ivar'][:, 0, :]
	ivar = data['data_ivar'][:, 1, :]
	wg = wave_grid(data)
	z_grid = lambda_to_z(wg) #Convert wavelength space to redshift space
	
	#sigma_v size same as number of Gaussian width models
	results = np.zeros((z_range.size, image.shape[0], sigma_v.size))
	
	#Save all the amplitudes to pass this to the PeakZoom function
	Amps = np.zeros((z_range.size, image.shape[0], sigma_v.size)) 
	
	for i, z in enumerate(z_range):
		wg2 = Window(z, wg, z_grid)
		widths = widthlist(z)
		print(widths[-1])
		model = Model(z, wg2, widths)
		
		#Find the idx of the edges of the windows and slice the image file to multiply with modelPrime
		minidx = np.where(wg == np.min(wg2))[0][0] 
		maxidx = np.where(wg == np.max(wg2))[0][0]
		imageSliced = image[:,minidx:maxidx+1]
		
		medians = np.median(imageSliced, axis = 1) #Median continuum subtraction
		imageSliced = imageSliced - medians[:, np.newaxis]
		
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
		Amps[i] = Amp
		
	results = results.transpose([1, 2, 0]) #This maintains the indices
	Amps = Amps.transpose([1, 2, 0])
	
	return z_range, widths, results, Amps

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

def plotterSNR2D(maskname, idx, z, widths, SNRdata):
	"""Returns redshift vs. width plot with SNR strength.
	
	Parameters
	----------
	maskname: name of the mask + '-' + grating number
	idx: index of a slit for a given maskname
	z: 0th output of the SNR_calculator function; redshift range
	widths: 1st output of the SNR_calculator function; width range
	SNRdata: 2nd output of the SNR_calculator function; SNR data cube
	
	Returns
	-------
	PDF of the 2D plot
	"""
	
	plt.imshow(SNRdata[idx], aspect='auto', interpolation='None', \
			extent=[np.min(z), np.max(z), np.min(widths), np.max(widths)], vmin=0)#, vmax=7)
	plt.colorbar()
	plt.ylabel('width', fontsize = 15, fontname = 'serif')
	plt.xlabel('redshift', fontsize = 15, fontname = 'serif')
	plt.title('Mask: ' + maskname + ', ' + 'Slit ' + str(idx),  fontsize = 15, fontname = 'serif')
	plt.savefig("results/SNR2D/"  + maskname + '/' + maskname + '-' + str(idx) + "-SNR2d.pdf", dpi = 600, bbox_inches = None)
	plt.close()
	
def SNRvz(maskname, idx, z, widths, SNRdata, Ampdata, image, ivar, wavelength_grid):
	"""Returns SNR vs z plot per slit and redshift and w values
	Parameters
	----------
	maskname: name of the mask + '-' + grating number
	idx: index of a slit for a given maskname
	z: 0th output of the SNR_calculator function; redshift range
	widths: 1st output of the SNR_calculator function; width range
	SNRdata: 2nd output of the SNR_calculator function; SNR data cube
	image: spectra 1d -> pass to PeakZoom
	ivar: inverse variance 1d -> pass to PeakZoom
	"""	
	
	def PeakZoom(maskname, idx, z, widths, SNRdata, Ampdata, image, ivar, wavelength_grid, w, redshift):
		"""Returns zoomed-in 1d spectra and inverse variance plots around the maxima redshift
		maskname: name of the mask + '-' + grating number
		idx: index of a slit for a given maskname
		z: 0th output of the SNR_calculator function; redshift range
		widths: 1st output of the SNR_calculator function; width range
		SNRdata: 2nd output of the SNR_calculator function; SNR data cube
		image: spectra 1d -> pass to PeakZoom
		ivar: inverse variance 1d -> pass to PeakZoom
		wavelength_grid: wavelength grid of the 1d spectra
		w: width indices of the best models
		redshift: redshift indices of the best models
		"""
		
		ranges = 20 #Arbitrary value
		
		imagetmp = image[idx, :]
		ivartmp = ivar[idx, :]
		
		#Corresponding wavelength
		lambda0 = 3728.4835 #From calculation
		
		wg = wavelength_grid[(wavelength_grid > (1+z[redshift])*(3728.4835-20.)) & (wavelength_grid < (1+z[redshift])*(3728.4835+20.))]
		f, axarr = plt.subplots(2, sharex=True)
		axarr[0].plot(wavelength_grid, imagetmp, c = 'k')
		axarr[0].set_title('Mask: ' + maskname + ', ' + 'Slit ' + str(idx) + "\n" 'Lambda = ' + str(np.round(lambda0*(1+z[redshift])-.1, 3))\
		, fontsize = 15, fontname = 'serif')
		axarr[1].plot(wavelength_grid, ivartmp, c = 'k')
		#axarr[1].set_title('1D inverse variance', fontsize = 15, fontname = 'serif')
		axarr[0].set_xlim([lambda0*(1+z[redshift])-ranges, lambda0*(1+z[redshift])+ranges])
		axarr[1].set_xlim([lambda0*(1+z[redshift])-ranges, lambda0*(1+z[redshift])+ranges])
		axarr[0].set_ylabel('spectra', fontsize = 15, fontname = 'serif')
		axarr[1].set_ylabel('inverse variance', fontsize = 15, fontname = 'serif')
		axarr[0].plot(wg, Model(z[redshift], wg, widths[w], Amp=Ampdata[idx, w, redshift]), c = 'red') #Plot the model
		
		plt.savefig("results/PeakZoom/" + maskname + '/' + maskname + '-' + str(idx) + "-zoom1d.pdf", dpi = 600, bbox_inches = None)
		plt.close()
	
	#Find width and z indices for highest SNR
	w, redshift = np.unravel_index(np.nanargmax(SNRdata[idx]), np.array(SNRdata[idx]).shape)
	
	if(SNRdata[idx, w, redshift] >= 7):
		#Generate SNR vs z plot per slit
		plt.plot(z, SNRdata[idx, w])
		plt.axhline(7, c = 'red') #Threshold of SNR = 7
		plt.ylabel('SNR', fontsize = 15, fontname = 'serif')
		plt.xlabel('redshift', fontsize = 15, fontname = 'serif')
		plt.title('Mask: ' + maskname + ', ' + 'Slit ' + str(idx) +"\n" +\
			"z = " + str(np.round(z[redshift], 3)) + ', w = ' + str(np.round(widths[w],2)) \
			, fontsize = 15, fontname = 'serif')
		#plt.xlim([z[redshift] - .1, z[redshift] + .1])
		plt.savefig("results/SNRvsRedshift/"  + maskname + '/' + maskname + '-' + str(idx) + "-SNR_vs_z.pdf", dpi = 600, bbox_inches = None)
		plt.close()
		
		PeakZoom(maskname, idx, z, widths, SNRdata, Ampdata, image, ivar, wavelength_grid, w, redshift)
		return z[redshift], widths[w]
	else:
		return np.nan, np.nan

"""------START OF JAE'S CODE------"""

def preprocess_bino(fname_data, fname_err, data_dir):
	"""
	Preprocessor goals.
	Data: 
		- If NaN: Data ZERO and Error infinity.
	Error:
		- If NaN: Error infinity and Data ZERO.
	Output: 
		- A numpy array of shape (Ntargets+1, 2, 32, num_cols). 
			- Though the native data have different number of rows, we use a single fixed number here.
			- Channel 0: Data
			- Channel 1: Error
		- List of headers

	The native data unit is ergs/cm^2/s/nm. Preprocessor changes this to
	10^-17 ergs/cm^2/s/Angstrom.

	First spectrum in native data is saved in loc "1". We follow the same convention.
	"""
	infinity = 1e60
	unit_conversion = 10**18

	# ---- Output variables
	data_err = None
	list_headers = [None]

	# ---- Python list of spectral data/err
	# Zeroth element is a blank.
	data = fits.open(data_dir + fname_data)
	err = fits.open(data_dir + fname_err)

	# ---- Place holder for the output array
	Ncols = data[1].data.shape[1]
	data_err = np.zeros((len(data), 2, 32, Ncols))
	data_err[:, 1, :, :] = infinity  # All data/errors are initially set to zero and infinity.

	for i in range(1, len(data)):
		# ---- Import data
		data_tmp = data[i].data * unit_conversion
		err_tmp = err[i].data * unit_conversion

		# ---- Apply preprocessing
		ibool = np.logical_or(np.isnan(err_tmp), np.isnan(data_tmp), err_tmp <=0.)
		data_tmp[ibool] = 0
		err_tmp[ibool] = infinity

		# ---- Trim the data
		idx_min, idx_max = index_edges(data_tmp)
		L_trim = 50
		data_tmp[:, :idx_min+L_trim] = 0
		data_tmp[:, idx_max-L_trim:] = 0
		err_tmp[:, :idx_min+L_trim] = infinity
		err_tmp[:, idx_max-L_trim:] = infinity


		# ---- Save data
		# Nrows = min(32, data_tmp.shape[0])
		# Data is usually crap outside this range
		data_err[i, 0, 4:25] = data_tmp[4:25]
		data_err[i, 1, 4:25] = err_tmp[4:25]

		# ---- Save header
		header_tmp = data[i].header
		list_headers.append(header_tmp)

	# # ---- Plot for bebugging 
	# vmin = -4
	# vmax = 4
	# plt.close()
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
	# # Plot data
	# data_plot = ax1.imshow(data_tmp, aspect="auto", cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
	# plt.colorbar(data_plot, ax = ax1)
	# # Plot err
	# err_plot = ax2.imshow(err_tmp, aspect="auto", cmap="gray", interpolation="none", vmin=0.02, vmax=0.05)
	# plt.colorbar(err_plot, ax = ax2)

	# plt.show()
	# plt.close()        
		
	return data_err, list_headers

def bit_from_header(header):
	name = header["SLITOBJ"]
	if name == "stars":
		name = 2**1
	elif name == "gal":
		name = 2**2
	return int(name)

def extract_single_data(data_err, list_headers, specnum):
	"""
	Extract single spectrum data, err, header from the list, arr provided by Preprocessor.
	- specnum: Target object number in a file. Ranges from 1 through approx. 140.
	"""
	header = list_headers[specnum]
	data = data_err[specnum, 0]
	err = data_err[specnum, 1]
	
	return data, err, header

def ivar_from_err(err):
	return 1./np.square(err)

def naive_profile(data, ivar, idx_min=0, idx_max=-1, L_trim = 500):
	"""
	Assumes D_ij ~ Norm(f_j * k_i, sig_ij) where f_j = f.
	"""
	if L_trim > 0:
		data = data[:, idx_min+L_trim:idx_max-L_trim]
		ivar = ivar[:, idx_min+L_trim:idx_max-L_trim]
	K = np.sum(data * ivar, axis = 1) / np.sum(ivar, axis = 1)
	K /= np.sum(K) # Normalization step.
	return K 

def produce_spec1D(data_err, list_headers, sig_K):
	"""
	Given 2D spectrum and the extraction kernel width sig_K,
	produce 1D spectra (Ntargets+1, 2, Ncols) and their inverse variance.
	"""
	data_ivar_1D = np.zeros((data_err.shape[0], 2, data_err.shape[3]))
	for specnum in range(1, len(list_headers)):
		data, err, header = extract_single_data(data_err, list_headers, specnum)
		ivar = ivar_from_err(err)

		spec1D_ivar = np.sum(np.square(K_T) * ivar, axis=0)
		spec1D = np.sum(K_T * data * ivar, axis=0) / spec1D_ivar
		
		data_ivar_1D[specnum, 0] = spec1D
		data_ivar_1D[specnum, 1] = spec1D_ivar
	return data_ivar_1D
		
def SIDE_from_header(header):
	return header["SIDE"]

def index_edges(data, num_thres=20):
	"""
	Given long postage stamp of data, return the edges.
	"""
	idx_min = 0
	idx_max = data.shape[1]-1
	tally = np.sum(data == 0., axis=0)
	while tally[idx_min] > num_thres:
		idx_min += 1
	while tally[idx_max] > num_thres:
		idx_max -=1
	return idx_min, idx_max

def gauss_fit2profile(K):
	# ---- Grid search for mu and sigma for the best Gaussian representation of the empirical kernel.
	Nrows = 32 
	mu_arr = np.arange(5, 20, 0.1)
	sig_arr = np.arange(1., 3., 0.05)
	chi_arr = np.zeros((mu_arr.size, sig_arr.size))
	x_arr = np.arange(0, Nrows, 1)
	for i, mu in enumerate(mu_arr):
		for j, sig in enumerate(sig_arr):
			A = np.exp(-np.square(x_arr-mu) / (2 * sig**2)) / (np.sqrt(2 * np.pi) * sig)
			chi_arr[i, j] = np.sum(np.square(K - A))
	# ---- Best fit
	idx = np.unravel_index(np.argmin(chi_arr), chi_arr.shape)
	mu_best = mu_arr[idx[0]] 
	sig_best = sig_arr[idx[1]]
	
	return mu_best, sig_best

def extract_stellar_profiles(data_err, list_headers):
	K_collection = []
	for specnum in range(1, len(list_headers)):
		data, err, header = extract_single_data(data_err, list_headers, specnum)
		ivar = ivar_from_err(err)

		BIT = bit_from_header(header)
		# ---- Perform optimal extraction 1D spectrum from 2D
		if (BIT == 2):
			idx_min, idx_max = index_edges(data)
			K = naive_profile(data, ivar, idx_min, idx_max, L_trim = 500)
			K_collection.append(K) # Collect K into a list.
	return K_collection
		
def remove_outlier(arr, std_thres = 2):
	"""
	Remove outliers in 1D array by sigma clipping.
	"""
	std = np.std(arr)
	mu = np.median(arr)
	
	return arr[(arr - mu) < (std_thres * std)]


def extraction_kernel_sig(K_collection):
	"""
	Based on the extracted stellar profiles, 
	compute a reasonable gaussian extraction kernal
	width (sigma).
	"""
	# Array of estimates gaussian means and widths
	K_gauss_mus = np.zeros(len(K_collection))
	K_gauss_sig = np.zeros(len(K_collection))
	for i in range(len(K_collection)):
		mu_best, sig_best = gauss_fit2profile(K_collection[i])    
		K_gauss_mus[i] = mu_best
		K_gauss_sig[i] = sig_best

	return np.median(K_gauss_sig)

def K_gauss_profile(mu, sig, Nrows = 32):
	"""
	Generate gaussian extraction profile of length Nrows
	given mu and sig.
	"""
	x_arr = np.arange(0, Nrows, 1)
	K_gauss = np.exp(-(x_arr - mu)**2 / (2 * sig**2))
	K_gauss /= np.sum(K_gauss)
	
	return K_gauss

def plot_kernels(K_collection, K_extract, fname):
	"""
	Plot the collection of stellar kernels and the ultimate
	extraction kernel at the center.
	"""

	fig, ax = plt.subplots(1, figsize=(10, 5))
	for i in range(len(K_collection)):
		ax.plot(K_collection[i], c="red", lw=0.5)    
	ax.plot(K_extract, c="blue", lw=1.5)
	ax.set_ylim([-0.03, 0.3])
	ax.axhline(y=0, c="black", ls="--", lw=1.)
	plt.savefig(fname, dpi=200, bbox_inches="tight")
#     plt.show()
	plt.close()

	return


def K_gauss_profile(mu, sig, Nrows = 32):
	"""
	Generate gaussian extraction profile of length Nrows
	given mu and sig.
	"""
	x_arr = np.arange(0, Nrows, 1)
	K_gauss = np.exp(-(x_arr - mu)**2 / (2 * sig**2))
	K_gauss /= np.sum(K_gauss)
	
	return K_gauss


def produce_spec1D(data_err, list_headers, sig_K, fname_prefix=None, verbose=True):
    """
    Given 2D spectrum and the extraction kernel width sig_K,
    produce 1D spectra (Ntargets+1, 2, Ncols) and their inverse variance.
    """
    Ncols = 32    
    data_ivar_1D = np.zeros((data_err.shape[0], 2, data_err.shape[3]))

    for specnum in range(1, len(list_headers)):
        if verbose and ((specnum % 10) == 0):
            print("Processing spec num: %d" % specnum)

        # ---- Extract the individual data
        data, err, header = extract_single_data(data_err, list_headers, specnum)
        ivar = ivar_from_err(err)

        # ---- Compute the center of the extraction
        idx_min, idx_max = index_edges(data)

        # ---- Algorithmically determine the row location of the spectra.
        # Note that I assume the center of the spectrum falls between 10 and 20.
        row_centers = []
        data_tmp = np.copy(data)
        ivar_tmp = np.copy(ivar)
        data_tmp[:10, :] = 0.
        data_tmp[20:, :] = 0.        
        ivar_tmp[:10, :] = 1e-120
        ivar_tmp[20:, :] = 1e-120        

        for idx in range(idx_min, idx_max-Ncols, Ncols//2):
            # Compute naive profile based on clipped 2D (32, 32) post stamps
            K = naive_profile(data_tmp[:, idx:idx+Ncols], ivar_tmp[:, idx:idx+Ncols], L_trim=-1)
            # Median filtering to reduce noise
            K = median_filter(K, size=5)
    #             # Savgol filtering of the naive profile         
    #             K_filtered = savgol_filter(K, window_length=9, polyorder=3)
            # Compute the center
            row_centers.append(np.argmax(K))

        # Compute the extraction profile using the above computed center and using extraction width.
        row_centers = np.asarray(row_centers)
        row_centers = row_centers[(row_centers > 9) & (row_centers < 21)]
        mu = np.round(np.median(row_centers))
        K_T = K_gauss_profile(mu, sig_K).reshape((K.size, 1))

        # ---- 1D extraction performed here
        spec1D_ivar = np.sum(np.square(K_T) * ivar, axis=0)
        spec1D = np.sum(K_T * data * ivar, axis=0) / spec1D_ivar

        # ---- Save the extracted spectrum
        data_ivar_1D[specnum, 0] = spec1D
        data_ivar_1D[specnum, 1] = spec1D_ivar        

        if fname_prefix is not None:
            plt.close()
            # ---- Spec figures
            fname = fname_prefix + "spec%d-2D.png" %specnum
            fig, ax = plt.subplots(1, figsize=(17, 1))
            ax.imshow(data, aspect="auto", cmap="gray", interpolation="none", vmin=-0.5, vmax=0.5)
            ax.axhline(y=mu+0.5, c="red", ls="--", lw=0.4)
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()

            # ---- Histogram of centers determined
            fname = fname_prefix + "spec%d-centers.png" %specnum
            fig, ax = plt.subplots(1, figsize=(7, 3))
            ax.hist(row_centers, bins=np.arange(0.5, 32.5, 1), histtype="step", color="black", normed=True)
            ax.plot(K_T, c="red", label="K_stellar")
            ax.axvline(x=mu, c="red", ls="--", lw=0.4)
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()

    return data_ivar_1D

def wavegrid_from_header(header, Ncols):
    """
    Construct a linear grid based on the header
    and a user specified number of columns.
    """
    x0 = header["CRVAL1"] * 10
    dx = header["CDELT1"] * 10
    return x0 + np.arange(0, Ncols, 1.) * dx
    