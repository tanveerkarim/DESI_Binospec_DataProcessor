from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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

def naive_profile(data, ivar):
	"""
	Assumes D_ij ~ Norm(f_j * k_i, sig_ij) where f_j = f.
	"""
	K = np.sum(data * ivar, axis = 1) / np.sum(ivar, axis = 1)
	K /= np.sum(K) # Normalization step.
	return K 


def extraction_kernel(data_err, list_headers, save_fig=None):
    """
    Given data_err array (Ntargets+1, 2, Nrows, Ncols) and headers, compute an extraction kernel
    using the following ad hoc recipe.
    - For each F-star observed, compute naive kernel using naive_profile function. 
    - K_combined: Find the maximum of the union of F-star naive profiles at each row pixel position.
    Also, any negative value is set equal to zero.
    - K_filtered: Apply savgol_filter with window_length=9 and polyorder=3 to smooth. Any negative value is 
    set equal to zero.
    - K_gauss: Based on K_filtered, find the best fitting gaussian using grid search method.
    - K_final: Maximum of the K_gauss and K_filtered.
    """
    K_collection = []
    K_combined = None
    for specnum in range(1, len(list_headers)):
        data, err, header = extract_single_data(data_err, list_headers, specnum)
        ivar = ivar_from_err(err)

        BIT = bit_from_header(header)
        # ---- Perform optimal extraction 1D spectrum from 2D
        if (BIT == 2):
            K = naive_profile(data, ivar)
            K_collection.append(K) # Collect K into a list.
        
    # ---- Combined kernel
    K_arr = np.zeros((len(K_collection), K.size))
    for i in range(len(K_collection)):
    	K_arr[i] = K_collection[i]
    K_combined = np.percentile(K_arr, 80, axis=0)
    K_combined /= np.sum(K_combined)    
    
    # ---- Filtered kernel
    K_filtered = savgol_filter(K_combined, window_length=9, polyorder=3)
    K_filtered = np.maximum(0, K_filtered)
    K_filtered /= np.sum(K_filtered) # Normalized the filter    
    
    
    # ---- Grid search for mu and sigma for the best Gaussian representation of the empirical kernel.
    Nrows = 32 
    mu_arr = np.arange(5, 20, 0.05)
    sig_arr = np.arange(3, 10, 0.05)
    chi_arr = np.zeros((mu_arr.size, sig_arr.size))
    x_arr = np.arange(0, Nrows, 1)
    for i, mu in enumerate(mu_arr):
        for j, sig in enumerate(sig_arr):
            A = np.exp(-np.square(x_arr-mu) / (2 * sig**2)) / (np.sqrt(2 * np.pi) * sig)
            chi_arr[i, j] = np.sum(np.square(K_filtered - A))
    # ---- Best fit
    idx = np.unravel_index(np.argmin(chi_arr), chi_arr.shape)
    mu_best = mu_arr[idx[0]] 
    sig_best = sig_arr[idx[1]] + 0.5 # Intentional broadening

    # ---- K_gauss
    K_gauss = np.exp(-np.square(x_arr-mu_best) / (2 * sig_best**2)) / (np.sqrt(2 * np.pi) * sig_best)

    # ---- K_final 
    K_final = np.minimum(K_gauss, K_filtered)
    K_final /= np.sum(K_final)
    
    if save_fig is not None:
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 10))
        # ---- K_collection and combined
        ax1.plot(range(K_combined.size), K_combined, c="blue", lw=2., label="K_combined")
        for i in range(len(K_collection)):
            ax1.plot(range(K_combined.size), K_collection[i], c="red", lw=0.5)
        ax1.set_ylim([-0.05, 0.3])
        ax1.legend(loc="upper right", fontsize=15)
        ax1.axhline(y=0., lw=1, ls="--", c="black")
        # ---- K_combined and filtered
        ax2.plot(range(K_combined.size), K_filtered, c="blue", lw=2., label="K_filtered")
        for i in range(len(K_collection)):
            ax2.plot(range(K_combined.size), K_collection[i], c="red", lw=0.5)        
        ax2.set_ylim([-0.05, 0.3])            
        ax2.axhline(y=0., lw=1, ls="--", c="black")
        ax2.legend(loc="upper right", fontsize=15)
        # ---- K_filtered and final
        ax3.plot(range(K_combined.size), K_gauss, c="black", lw=2., label="K_gauss")
        ax3.plot(range(K_combined.size), K_final, c="blue", lw=2., label="K_final")
        for i in range(len(K_collection)):
            ax3.plot(range(K_combined.size), K_collection[i], c="red", lw=0.5)                
        ax3.axhline(y=0., lw=1, ls="--", c="black")
        ax3.set_ylim([-0.05, 0.3])        
        ax3.legend(loc="upper right", fontsize=15)

        plt.savefig(save_fig, dpi=200, bbox_inches = "tight")
        plt.close()

    return K_collection, K_combined, K_filtered, K_gauss, K_final
        

def produce_spec1D(data_err, list_headers, K):
    """
    Given 2D spectrum and the extraction kernel K,
    produce 1D spectra (Ntargets+1, 2, Ncols) and their inverse variance.
    """
    K_T = K.reshape((K.size, 1))
    data_ivar_1D = np.zeros((data_err.shape[0], 2, data_err.shape[3]))
    for specnum in range(1, len(list_headers)):
        data, err, header = extract_single_data(data_err, list_headers, specnum)
        ivar = ivar_from_err(err)
        spec1D_ivar = np.sum(np.square(K_T) * ivar, axis=0)
        spec1D = np.sum(K_T * data * ivar, axis=0) / spec1D_ivar
        
        data_ivar_1D[specnum, 0] = spec1D
        data_ivar_1D[specnum, 1] = spec1D_ivar
    return data_ivar_1D
        