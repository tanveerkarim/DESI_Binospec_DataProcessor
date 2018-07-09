import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from specz_extractor_1D import lambda_to_z, Window, Model

"""------GENERATE WAVELENGTH GRID SAME AS MMT BINOSPEC------"""
data_dir = "../../../DATA_MAY18/spec1d/"
fname270 = '2-8h30m-270-spec1d.npz'
fname600 = '2-8h30m-600-spec1d.npz'
data270 = np.load(data_dir + fname270)
data600 = np.load(data_dir + fname600)

def wave_grid(data):
    """Returns wavegrid based on header file from data"""
    
    crval1 = float(str(data['headers'][1]).split("CRVAL1")[1].split("=")[1].split("/")[0]) #Starting value
    cdelt1 = float(str(data['headers'][1]).split("CDELT1")[1].split("=")[1].split("/")[0]) #Pixel size
    
    collapsedSpectrum = data['data_ivar'][:, 0, :]
    
    wave_grid = crval1 + cdelt1 * np.arange(collapsedSpectrum[1].shape[0])
    wave_grid *= 10 #Convert wave_grid to Angstrom from nm
    return wave_grid
	
wg270 = wave_grid(data270)
wg600 = wave_grid(data600)
"""------------"""
#User defines which grating to generate
wg_key = int(input("Enter grating number: "))
ngal = int(input("Enter number of galaxies: "))

if(wg_key == 270):
	wg = wg270
elif(wg_key == 600):
	wg = wg600
else:
	print("Enter 270 or 600.")

"""------FUNCTIONS TO GENERATE MOCK 1D CATALOGUES------"""
def ModelGen(wg, z, width, Amp = 1, offset = 0, GaussianNoiseTrue = False):
    """Generates mock Gaussian filter model at redshift z with an optional offset
    
    Parameters
    ----------
    z: array of redshifts at which the model is being tested
    wg: pixel grid of the Window
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

    if(not GaussianNoiseTrue == True):
        model = (Amp/2*(Gaussian(wg, lambda_obs - separation_r, width) \
                    + Gaussian(wg, lambda_obs + separation_r, width))) + offset
        return model
    else:
        errstd = 0.001
        errarray = np.random.normal(0, errstd, wg.shape)[:, np.newaxis]
        model = (Amp/2*(Gaussian(wg, lambda_obs - separation_r, width) \
                    + Gaussian(wg, lambda_obs + separation_r, width))) \
             + offset + errarray
        return model, 1/errarray**2
		
def mockCatalogue(maskgrating, ngal = 1, seed = 100):
    """Generates 1d mock masks as MMT"""
    
    if(maskgrating == 270):
        zlow = 0.0; zhigh = 1.5
        wg = wg270
    elif(maskgrating == 600):
        zlow = 0.7; zhigh = 1.61
        wg = wg600
        
    widths = np.arange(0.1, 2, 0.05)
    
    models = np.zeros((ngal, len(wg)))
    ivars = np.zeros((ngal, len(wg)))
    zlist = np.zeros(ngal)
    wlist = np.zeros(ngal)
    offsetlist = np.zeros(ngal)
    
    np.random.seed(seed)
    for i in range(ngal):
        z = np.round(np.random.uniform(zlow, zhigh, 1), 3)
        w = np.round(np.random.uniform(0.1, 2., 1),2)
        offset = np.round(np.random.uniform(0.0, 2., 1),1)
        
        tmp1, tmp2 = ModelGen(z = z, width = w, wg = wg, offset = offset,\
                      GaussianNoiseTrue = True)
        
        models[i] = tmp1.reshape((len(tmp1),))
        ivars[i] = tmp2.reshape((len(tmp2),))
        zlist[i] = z
        wlist[i] = w
        offsetlist[i] = offset
        
    #injectionModels = np.vstack(zlist, wlist, offsetlist) #Injection model matrix is z, w, offset respectively
    #injectionModels = injectionModels.T #Transpose for readability
    
    #To make the data structure look same as Jae's .npz files
    tmp = np.dstack((models, ivars))
    tmp = np.transpose(tmp, (0,2,1))
    
    np.savez("mock1D_data", data_ivar=tmp)
    np.savez("mock1D_models", z=zlist, w=wlist, offset=offsetlist)
"""------------"""
"""------------"""

"""------MOCK GENERATION AND ANALYSIS------"""
mockCatalogue(wg_key, ngal = ngal)


