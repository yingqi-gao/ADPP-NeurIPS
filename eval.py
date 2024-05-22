import random
from functools import partial
from _pricing_utils import get_epc_rev
import numpy as np
from _pricing_mechanisms import DOP, RSOP, RSKDE
import dill
from _util import get_regrets
import sys



def eval(dist_type: str, mech: str, num_training_bids: int | None, num_training_rounds: int | None):
    with open("./sim/" + dist_type + "/tests.pkl", "rb") as f:
        tests = dill.load(f)
    test_bids = tests["test_bids"]
    ideal_revenue = tests["ideal_revenue"]
    true_dist_cdf = tests["true_dist_cdf"]
    if mech == "DOP":
        DOP_regrets = []
        for i in range(200):
            DOP_price = DOP(dict(zip(range(10), test_bids[i])))
            regret = ideal_revenue - get_epc_rev(DOP_price, value_cdf = true_dist_cdf)
            DOP_regrets.append(regret)
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/" + mech + "_regrets.pkl", "wb") as f:
            dill.dump(DOP_regrets, f)
    elif mech == "RSOP":
        RSOP_regrets = []
        for i in range(200):
            regrets = []
            for seed in range(200):
                RSOP_price = RSOP(dict(zip(range(10), test_bids[i])), random_seed = seed)
                regret = ideal_revenue - get_epc_rev(RSOP_price, value_cdf = true_dist_cdf)
                regrets.append(regret)
            RSOP_regrets.append(np.mean(regrets))
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/" + mech + "_regrets.pkl", "wb") as f:
            dill.dump(RSOP_regrets, f)
    elif mech == "RSKDE":
        RSKDE_regrets = []
        for i in range(200):
            regrets = []
            for seed in range(200):
                RSKDE_price, _ = RSKDE(dict(zip(range(10), test_bids[i])), lower = 1, upper = 10, random_seed = seed)
                regret = ideal_revenue - get_epc_rev(RSKDE_price, value_cdf = true_dist_cdf)
                regrets.append(regret)
            RSKDE_regrets.append(np.mean(regrets))
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/" + mech + "_regrets.pkl", "wb") as f:
            dill.dump(RSKDE_regrets, f)
    elif mech == "RSRDE":
        get_regrets(dist_type, num_training_bids, num_training_rounds, true_dist_cdf, ideal_revenue, test_bids)




def main():
    dist_type = sys.argv[1]
    mech = sys.argv[2]
    num_training_bids = int(sys.argv[3])
    num_training_rounds = int(sys.argv[4])
    eval(dist_type, mech, num_training_bids, num_training_rounds)



if __name__ == "__main__":
    main()