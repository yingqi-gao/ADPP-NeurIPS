import random
from argparse import ArgumentParser
import scipy
import pandas as pd
import numpy as np
from _py_density_estimation import get_bw, rde_training_py
import pickle
import os
import sys

from _util import pickle_dump, pickle_load, REAL_DATA_PKL_PATH, PARTITION_REPETITIONS


def train(dist_type: str, output_uri: str):
    train_bids = []
    lower = 1
    upper = 10
    if dist_type == "real":
        real_data_pkl = pickle_load(REAL_DATA_PKL_PATH) 
        lower = 6.9
        upper = 20.6

    for seed in range(PARTITION_REPETITIONS):
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
            train_bids_pr = random.choices(real_data_pkl.tolist(), k = 1000)
        else:
            raise ValueError("Unidentified/unsupported distribution types.")
        train_bids.append(train_bids_pr)
        
    training_results_100_20 = rde_training_py(train_hist = [bids[:100] for bids in train_bids[:20]],
                                              train_bws = [get_bw(bids[:100]) for bids in train_bids[:20]],
                                              lower = lower, upper = upper)
    # training_results_100_200 = rde_training_py(train_hist = [bids[:100] for bids in train_bids],
    #                                            train_bws = [get_bw(bids[:100]) for bids in train_bids],
    #                                            lower = lower, upper = upper)
    training_results_1000_20 = rde_training_py(train_hist = [bids for bids in train_bids[:20]],
                                               train_bws = [get_bw(bids) for bids in train_bids[:20]],
                                               lower = lower, upper = upper)
    # training_results_1000_200 = rde_training_py(train_hist = [bids for bids in train_bids],
    #                                             train_bws = [get_bw(bids) for bids in train_bids],
    #                                             lower = lower, upper = upper)
    path = os.path.join(output_uri, dist_type, "trains.pkl")
    pickle_dump([training_results_100_20, training_results_1000_20], path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dist-type", required=True, help="Specify the distribution family or 'real'.", choices=["uniform", "normal", "exponential", "real"])
    parser.add_argument("--output-uri", default="./sim/")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.dist_type, args.output_uri)


if __name__ == "__main__":
    main()