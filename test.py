from argparse import ArgumentParser
from functools import partial
import os
import random

import scipy

from _pricing_utils import max_epc_rev
from _py_density_estimation import kde_py




from _util import pickle_dump, pickle_load, REAL_DATA_PKL_PATH, TEST_REPETITIONS


def test(dist_type: str, output_uri: str):
    random.seed(0)
    if dist_type == "uniform":
        true_dist_cdf = partial(scipy.stats.uniform.cdf, **{"loc": 1, "scale": 10 - 1})
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 1, upper = 10)
        test_bids = []
        for seed in range(TEST_REPETITIONS):
            random.seed(seed)
            test_bids.append(scipy.stats.uniform.rvs(**{"loc": 1, "scale": 10 - 1}, size = 10).tolist())
    elif dist_type == "normal":
        true_mean = random.uniform(0, 20)
        true_sd = random.uniform(0, 10)
        true_dist_cdf = partial(scipy.stats.truncnorm.cdf, 
                                **{"a": (1 - true_mean) / true_sd,
                                   "b": (10 - true_mean) / true_sd,
                                   "loc": true_mean,
                                   "scale": true_sd})
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 1, upper = 10)
        test_bids = []
        for seed in range(TEST_REPETITIONS):
            random.seed(seed)
            test_bids.append(scipy.stats.truncnorm.rvs(**{"a": (1 - true_mean) / true_sd,
                                                        "b": (10 - true_mean) / true_sd,
                                                        "loc": true_mean,
                                                        "scale": true_sd}, 
                                                       size = 10).tolist())
    elif dist_type == "exponential":
        true_scale = random.uniform(0, 20)
        true_dist_cdf = partial(scipy.stats.truncexpon.cdf, 
                                **{"b": (9 - 1) / true_scale,
                                   "loc": 1,
                                   "scale": true_scale})
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 1, upper = 10)
        test_bids = []
        for seed in range(TEST_REPETITIONS):
            random.seed(seed)
            test_bids.append(scipy.stats.truncexpon.rvs(**{"b": (9 - 1) / true_scale,
                                                           "loc": 1,
                                                           "scale": true_scale}, 
                                                        size = 10).tolist())
    elif dist_type == "real":
        training_data = pickle_load(REAL_DATA_PKL_PATH)
        true_dist_cdf = kde_py(training_data.tolist(), lower = 6.9, upper = 20.6)
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 6.9, upper = 20.6)
        test_bids = []
        for seed in range(TEST_REPETITIONS):
            test_bids.append(random.choices(training_data.tolist(), k = 10))
    else:
        raise ValueError("Unidentified/unsupported distribution types.")
    
    path = os.path.join(output_uri, dist_type, "tests.pkl")
    pickle_dump(
        {
            "test_bids": test_bids,
            "ideal_revenue": ideal_revenue,
            "true_dist_cdf": true_dist_cdf
        },
        path
    )
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dist-type", required=True, help="Specify the distribution family or 'real'.", choices=["uniform", "normal", "exponential", "real"])
    parser.add_argument("--output-uri", default="./sim/")
    parser.add_argument("--r-home", required=True, help="Get your R_HOME path via running .libPaths() in R terminal.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['R_HOME'] = args.r_home
    test(args.dist_type, args.output_uri)



if __name__ == "__main__":
    main()       

    

    