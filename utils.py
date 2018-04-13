from astropy.io import fits
import numpy as np

def bino_data_preprocess(data_fname, err_fname):
    """
    Import data and its errors and return a list of length (Nobjs+1), where each element
    corresponds to an image of (num_rows, num_cols, 2): the first channel correspond to the image,
    and the second to the errors. 

    +1 is for the first empty HDU. num_rows and num_cols vary over different spectra. 
    
    Note that error is set to infinity (i.e., 1e30) wherever NaN appears in the image.

	Args:
		- data_fname: File address for the image data.
		- err_fname: File address for the error.

    """
    #---- Loading in the 2D data for St82-1hr
    data2D = fits.open(data_fname)
    data2D_err = fits.open(err_fname)

    #---- Generate the place holder list for numpy array of both data and errors
    Nobjs = len(data2D)-1
    data = [None] #
    header = [None]

    #---- Loop through the imported data. Multiply by a common factor 1e16 to adjust the overall scale.
    for i in range(1, Nobjs+1):
        im = np.copy(data2D[i].data) * 1e16
        err = np.copy(data2D_err[i].data) * 1e16
        num_rows, num_cols = im.shape
        
        # Find pixels in the image or error that has NaN values
        iNaN = np.logical_or(np.isnan(im), np.isnan(err), err==0)
        
        # Set the NaN values equal to zero in the image and error to the infinity.
        im[iNaN] = 0
        err[iNaN] = 1e30
        
        # Properly save
        data_tmp = np.zeros((num_rows, num_cols, 2))
        data_tmp[:, :, 0] = im
        data_tmp[:, :, 1] = err
        
        data.append(data_tmp)
        header.append(data2D[i].header)
        
    return data, header