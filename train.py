import random
import scipy
import pandas as pd
import numpy as np
from _py_density_estimation import get_bw, rde_training_py
import pickle
import sys



def train(dist_type: str):
    train_bids = []
    for seed in range(200):
        random.seed(seed)
        if dist_type == "uniform":
            train_bids_pr = scipy.stats.uniform.rvs(**{"loc": 1, "scale": 10 - 1}, size = 1000).tolist()
        elif dist_type == "normal":
            sim_mean = random.uniform(0, 20)
            sim_sd = random.uniform(0, 10)
            train_bids_pr = scipy.stats.truncnorm.rvs(**{"a": (1 - sim_mean) / sim_sd,
                                                         "b": (10 - sim_mean) / sim_sd,
                                                         "loc": sim_mean,
                                                         "scale": sim_sd}, 
                                                      size = 1000).tolist()
        elif dist_type == "exponential":
            sim_scale = random.uniform(0, 20)
            train_bids_pr = scipy.stats.truncexpon.rvs(**{"b": (9 - 1) / sim_scale,
                                                          "loc": 1,
                                                          "scale": sim_scale}, 
                                                       size = 1000).tolist()
        elif dist_type == "real":
            df = pd.read_csv("./assets/auction97.csv")
            useful_df = df[["round", "item_name", "block", "bidder_name", "bid_amount"]].dropna()
            clean_df = useful_df.sort_values("round", ascending = False).groupby(["item_name", "bidder_name"]).first().reset_index()
            training_data = np.log(clean_df["bid_amount"][clean_df["block"] == "G"])
            train_bids_pr = random.choices(training_data.tolist(), k = 1000)
        else:
            raise ValueError("Unidentified/unsupported distribution types.")
        train_bids.append(train_bids_pr)
        
    training_results_100_20 = rde_training_py(train_hist = [bids[:100] for bids in train_bids[:20]],
                                              train_bws = [get_bw(bids[:100]) for bids in train_bids[:20]],
                                              lower = 1, upper = 10)
    training_results_100_200 = rde_training_py(train_hist = [bids[:100] for bids in train_bids],
                                               train_bws = [get_bw(bids[:100]) for bids in train_bids],
                                               lower = 1, upper = 10)
    training_results_1000_20 = rde_training_py(train_hist = [bids for bids in train_bids[:20]],
                                               train_bws = [get_bw(bids) for bids in train_bids[:20]],
                                               lower = 1, upper = 10)
    training_results_1000_200 = rde_training_py(train_hist = [bids for bids in train_bids],
                                                train_bws = [get_bw(bids) for bids in train_bids],
                                                lower = 1, upper = 10)
    
    with open("./sim/" + dist_type + "/trains.pkl", "wb") as f:
        pickle.dump([training_results_100_20, training_results_100_200, training_results_1000_20, training_results_1000_200], f)



def main():
    dist_type = sys.argv[1]
    train(dist_type)



if __name__ == "__main__":
    main()