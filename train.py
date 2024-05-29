from argparse import ArgumentParser
import os
import random

import scipy

from _py_density_estimation import get_bw, rde_training_py
from _util import get_train_uri, pickle_dump, pickle_load, set_seed, REAL_DATA_PKL_PATH


def train(
    *,
    dist_type: str,
    num_training_bids: int | None = None,
    num_training_rounds: int | None = None,
    output_uri: str,
):
    train_bids = []
    lower = 1
    upper = 10
    if dist_type == "real":
        real_data_pkl = pickle_load(REAL_DATA_PKL_PATH)
        lower = 6.9
        upper = 20.6

    for seed in range(1, num_training_rounds + 1):
        set_seed(seed)
        if dist_type == "uniform":
            train_bids_pr = scipy.stats.uniform.rvs(
                **{"loc": 1, "scale": 10 - 1}, size=num_training_bids
            ).tolist()
        elif dist_type == "normal":
            sim_mean = random.uniform(0, 20)
            sim_sd = random.uniform(0, 10)
            train_bids_pr = scipy.stats.truncnorm.rvs(
                **{
                    "a": (1 - sim_mean) / sim_sd,
                    "b": (10 - sim_mean) / sim_sd,
                    "loc": sim_mean,
                    "scale": sim_sd,
                },
                size=num_training_bids,
            ).tolist()
        elif dist_type == "exponential":
            sim_scale = random.uniform(0, 20)
            train_bids_pr = scipy.stats.truncexpon.rvs(
                **{"b": (9 - 1) / sim_scale, "loc": 1, "scale": sim_scale},
                size=num_training_bids,
            ).tolist()
        elif dist_type == "real":
            train_bids_pr = random.choices(real_data_pkl.tolist(), k=num_training_bids)
        else:
            raise ValueError("Unidentified/unsupported distribution types.")
        train_bids.append(train_bids_pr)

    training_results = rde_training_py(
        train_hist=train_bids,
        train_bws=[get_bw(bids) for bids in train_bids],
        lower=lower,
        upper=upper,
    )

    train_uri = get_train_uri(
        output_uri=output_uri,
        dist_type=dist_type,
        num_training_bids=num_training_bids,
        num_training_rounds=num_training_rounds,
    )
    pickle_dump(training_results, train_uri)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dist-type",
        required=True,
        help="Specify the distribution family or 'real'.",
        choices=["uniform", "normal", "exponential", "real"],
    )
    parser.add_argument("--num-training-bids", required=True, type=int)
    parser.add_argument("--num-training-rounds", required=True, type=int)
    parser.add_argument("--output-uri", default="./sim/")
    return parser.parse_args()


def main():
    args = parse_args()
    dist_type = args.dist_type
    num_training_bids = args.num_training_bids
    num_training_rounds = args.num_training_rounds
    train(
        dist_type=dist_type,
        num_training_bids=num_training_bids,
        num_training_rounds=num_training_rounds,
        output_uri=args.output_uri,
    )


if __name__ == "__main__":
    main()
