import random
from functools import partial
import scipy
from _pricing_utils import max_epc_rev
from _py_density_estimation import kde_py
import pandas as pd
import numpy as np
import dill
import sys



def test(dist_type: str):
    random.seed(0)
    if dist_type == "uniform":
        true_dist_cdf = partial(scipy.stats.uniform.cdf, **{"loc": 1, "scale": 10 - 1})
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 1, upper = 10)
        test_bids = []
        for seed in range(200):
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
        for seed in range(200):
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
        for seed in range(200):
            random.seed(seed)
            test_bids.append(scipy.stats.truncexpon.rvs(**{"b": (9 - 1) / true_scale,
                                                           "loc": 1,
                                                           "scale": true_scale}, 
                                                        size = 10).tolist())
    elif dist_type == "real":
        df = pd.read_csv("./assets/auction97.csv")
        useful_df = df[["round", "item_name", "block", "bidder_name", "bid_amount"]].dropna()
        clean_df = useful_df.sort_values("round", ascending = False).groupby(["item_name", "bidder_name"]).first().reset_index()
        training_data = np.log(clean_df["bid_amount"][clean_df["block"] == "G"])
        true_dist_cdf = kde_py(training_data.tolist(), lower = 6.9, upper = 20.6)
        ideal_price, ideal_revenue = max_epc_rev(true_dist_cdf, lower = 6.9, upper = 20.6)
        test_bids = []
        for seed in range(200):
            test_bids.append(random.choices(training_data.tolist(), k = 10))
    else:
        raise ValueError("Unidentified/unsupported distribution types.")
    
    with open("./sim/" + dist_type + "/tests.pkl", "wb") as f:
        dill.dump({"test_bids": test_bids,
                   "ideal_revenue": ideal_revenue,
                   "true_dist_cdf": true_dist_cdf}, f)
    
    
    
def main():
    dist_type = sys.argv[1]
    test(dist_type)



if __name__ == "__main__":
    main()       

    

    