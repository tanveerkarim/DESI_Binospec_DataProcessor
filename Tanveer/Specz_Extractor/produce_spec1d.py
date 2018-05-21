from utils import *

profile_figure_dir = "./kernels/"
mask_dirs = os.listdir("../../data/")
fname_data = "obj_abs_slits_lin.fits"
fname_err = "obj_abs_err_slits_lin.fits"

for mask in mask_dirs:
    if mask.endswith("0"):
        print("/-----", mask)        
        data_dir = "../../data/" + mask + "/"
        # ---- Import data
        data_err, list_headers = preprocess_bino(fname_data, fname_err, data_dir)

        # ---- Compute extraction kernel width
        K_collection = extract_stellar_profiles(data_err, list_headers)
        sig_extract = extraction_kernel_sig(K_collection) * 1.5 # Expand by 1.5
        K_extract = K_gauss_profile(15., sig_extract) 
        plot_kernels(K_collection, K_extract, "./kernels/"+mask+"-kernel.png")        

        # ---- Perform 1D extraction
        fname_prefix = "./spec1d/figures/" + mask +"/"
        if not os.path.isdir(fname_prefix):
            os.mkdir(fname_prefix)
        data_ivar_1D = produce_spec1D(data_err, list_headers, sig_extract, fname_prefix=fname_prefix)
        np.savez("./spec1d/" + mask +"-spec1d.npz", data_ivar = data_ivar_1D, headers = list_headers)