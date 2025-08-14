import sys
import pickle

def load_comparison_data(less_windows=True):
    if less_windows:
        path_all_fit_results = "./comparison_data/all_fit_results_short_v2.pkl"
    else:
        path_all_fit_results = "./comparison_data/all_fit_results_long_v2.pkl"


    with open(path_all_fit_results, "rb") as f:
        compare_all_fit_results = pickle.load(f)
    return compare_all_fit_results
