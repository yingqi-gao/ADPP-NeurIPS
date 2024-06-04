from argparse import ArgumentParser
from typing import Callable
import os

import numpy as np

from _pricing_utils import get_epc_rev
from _pricing_mechanisms import DOP, RSOP, RSKDE, RSRDE
from _util import (
    pickle_load,
    pickle_dump,
    get_train_uri,
    get_regrets_uri,
    TEST_REPETITIONS,
    PARTITION_REPETITIONS,
)


def get_RSRDE_regrets(
    *,
    dist_type: str,
    num_training_bids: int,
    num_training_rounds: int,
    true_dist_cdf: Callable,
    ideal_revenue: float,
    test_bids: list,
    output_uri: str,
    regrets_uri: str,
):
    # load training results
    training_result = pickle_load(
        get_train_uri(
            output_uri=output_uri,
            dist_type=dist_type,
            num_training_bids=num_training_bids,
            num_training_rounds=num_training_rounds,
        )
    )

    # check if regrets file exists
    if os.path.isfile(regrets_uri):
        regrets_means = pickle_load(regrets_uri)
    else:
        regrets_means = dict.fromkeys(range(TEST_REPETITIONS))

    if dist_type == "real":
        lower = 6.9
        upper = 20.6
    else:
        lower = 1
        upper = 10

    # get regrets
    for key, value in regrets_means.items():
        if value is None:
            regrets = []
            for seed in range(PARTITION_REPETITIONS):
                price, _ = RSRDE(
                    dict(zip(range(10), test_bids[key])),
                    lower=lower,
                    upper=upper,
                    random_seed=seed,
                    training_results=training_result,
                )
                regret = ideal_revenue - get_epc_rev(price, value_cdf=true_dist_cdf)
                regrets.append(regret)
            regrets_means[key] = np.mean(regrets)
            pickle_dump(regrets_means, regrets_uri)
        print(f"Done with round {key}")


def eval(
    *,
    dist_type: str,
    mechanism: str,
    num_training_bids: int | None = None,
    num_training_rounds: int | None = None,
    output_uri: str,
):
    lower = 1
    upper = 10
    if dist_type == "real":
        lower = 6.9
        upper = 20.6
        
    tests = pickle_load(os.path.join(output_uri, dist_type, "tests.pkl"))
    regrets_uri = get_regrets_uri(
        output_uri=output_uri,
        dist_type=dist_type,
        mechanism=mechanism,
        num_training_bids=num_training_bids,
        num_training_rounds=num_training_rounds,
    )
    test_bids = tests["test_bids"]
    ideal_revenue = tests["ideal_revenue"]
    true_dist_cdf = tests["true_dist_cdf"]
    if mechanism == "DOP":
        DOP_regrets = []
        for i in range(TEST_REPETITIONS):
            DOP_price = DOP(dict(zip(range(10), test_bids[i])))
            regret = ideal_revenue - get_epc_rev(DOP_price, value_cdf=true_dist_cdf)
            DOP_regrets.append(regret)
        pickle_dump(DOP_regrets, regrets_uri)
    elif mechanism == "RSOP":
        RSOP_regrets = []
        for i in range(TEST_REPETITIONS):
            regrets = []
            for seed in range(PARTITION_REPETITIONS):
                RSOP_price = RSOP(dict(zip(range(10), test_bids[i])), random_seed=seed)
                regret = ideal_revenue - get_epc_rev(
                    RSOP_price, value_cdf=true_dist_cdf
                )
                regrets.append(regret)
            RSOP_regrets.append(np.mean(regrets))
        pickle_dump(RSOP_regrets, regrets_uri)
    elif mechanism == "RSKDE":
        RSKDE_regrets = []
        for i in range(TEST_REPETITIONS):
            regrets = []
            for seed in range(PARTITION_REPETITIONS):
                RSKDE_price, _ = RSKDE(
                    dict(zip(range(10), test_bids[i])),
                    lower=lower,
                    upper=upper,
                    random_seed=seed,
                )
                regret = ideal_revenue - get_epc_rev(
                    RSKDE_price, value_cdf=true_dist_cdf
                )
                regrets.append(regret)
            RSKDE_regrets.append(np.mean(regrets))
        pickle_dump(RSKDE_regrets, regrets_uri)
    elif mechanism == "RSRDE":
        get_RSRDE_regrets(
            dist_type=dist_type,
            num_training_bids=num_training_bids,
            num_training_rounds=num_training_rounds,
            true_dist_cdf=true_dist_cdf,
            ideal_revenue=ideal_revenue,
            test_bids=test_bids,
            output_uri=output_uri,
            regrets_uri=regrets_uri,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dist-type",
        required=True,
        help="Specify the distribution family or 'real'.",
        choices=["uniform", "normal", "exponential", "real"],
    )
    parser.add_argument(
        "--mechanism",
        required=True,
        help="Specify the mechanism",
        choices=["DOP", "RSOP", "RSKDE", "RSRDE"],
    )
    parser.add_argument("--num-training-bids", type=int)
    parser.add_argument("--num-training-rounds", type=int)
    parser.add_argument("--output-uri", default="./sim/")
    parser.add_argument(
        "--r-home",
        required=True,
        help="Get your R_HOME path via running .libPaths() in R terminal.",
    )
    args = parser.parse_args()
    if args.mechanism == "RSRDE":
        if not args.num_training_bids or not args.num_training_rounds:
            raise ValueError(
                "Mechanism RSRDE must specify --num-training-bids and --num-training-rounds from choices."
            )
    return args


def main():
    args = parse_args()
    os.environ["R_HOME"] = args.r_home
    dist_type = args.dist_type
    mechanism = args.mechanism
    num_training_bids = args.num_training_bids
    num_training_rounds = args.num_training_rounds
    eval(
        dist_type=dist_type,
        mechanism=mechanism,
        num_training_bids=num_training_bids,
        num_training_rounds=num_training_rounds,
        output_uri=args.output_uri,
    )


if __name__ == "__main__":
    main()
