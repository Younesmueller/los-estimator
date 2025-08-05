import sys
import pickle

def load_comparison_data(less_windows=True,new=False):
    if not new:
        sys.path.append(r"C:\data\src\los-estimator\los-estimator\03_los_deconvolution")
        import los_fitter2
        import core as core
        sys.path.remove(r"C:\data\src\los-estimator\los-estimator\03_los_deconvolution")

        if less_windows:
            path_all_fit_results = "./comparison_data/all_fit_results_short.pkl"
        else:
            path_all_fit_results = "./comparison_data/all_fit_results_long.pkl"
    else:
        if less_windows:
            path_all_fit_results = "./comparison_data/all_fit_results_short_v2.pkl"
        else:
            path_all_fit_results = "./comparison_data/all_fit_results_long_v2.pkl"


    with open(path_all_fit_results, "rb") as f:
        compare_all_fit_results = pickle.load(f)
    return compare_all_fit_results
