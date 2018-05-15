from utils import *
import os
profile_figure_dir = "./kernels/"
mask_dirs = os.listdir("../../data/")
fname_data = "obj_abs_slits_lin.fits"
fname_err = "obj_abs_err_slits_lin.fits"

for mask in mask_dirs:
    print(mask)
    if mask.endswith("0"):
        data_dir = "../../data/" + mask + "/"
        # ---- Import data
        data_err, list_headers = preprocess_bino(fname_data, fname_err, data_dir)

        # ---- Compute extraction kernel
        K_collection, K_combined, K_filtered, K_gauss, K_final \
            = extraction_kernel(data_err, list_headers, profile_figure_dir+ mask + "-kernel.png")

        # ---- Perform 1D extraction
        data_ivar_1D = produce_spec1D(data_err, list_headers, K_final)
        np.save("./spec1d/" + mask +"-spec1d.npy", data_ivar_1D)
