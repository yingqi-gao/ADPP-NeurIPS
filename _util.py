import pickle
from typing import Callable
from _pricing_mechanisms import RSRDE
from _pricing_utils import get_epc_rev
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


def get_regrets(dist_type: str, num_training_bids: int, num_training_rounds: int,
                true_dist_cdf: Callable, ideal_revenue: float, test_bids: list):
    # load training results
    with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/trains.pkl", "rb") as f:
        training_results = pickle.load(f)
    with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets_" + str(num_training_bids)  + "_" + str(num_training_rounds) + ".pkl", "rb") as f:
        regret_means = pickle.load(f)
    if dist_type == "real":
        lower = 6.9
        upper = 20.6
    else:
        lower = 1
        upper = 10
    
    if num_training_bids == 100 and num_training_rounds == 20:
        training_result = training_results[0]
    elif num_training_bids == 100 and num_training_rounds == 200:
        training_result = training_results[1]
    elif num_training_bids == 1000 and num_training_rounds == 20:
        training_result = training_results[2]
    elif num_training_bids == 1000 and num_training_rounds == 200:
        training_result = training_results[3]
    
    # get regrets
    for key, value in regret_means.items():
        if value is None:
            regrets = []
            for seed in range(200):
                price, _ = RSRDE(dict(zip(range(10), test_bids[key])), 
                                 lower = lower, upper = upper, 
                                 random_seed = seed, 
                                 training_results = training_result)
                regret = ideal_revenue - get_epc_rev(price, value_cdf = true_dist_cdf)
                regrets.append(regret)
            regret_means[key] = np.mean(regrets)
            if dist_type == "real":
                with open("/u/scratch/y/yqg36/ADPP/data/real/regrets_" + str(num_training_bids) + "_" + str(num_training_rounds) + ".pkl", "wb") as f:
                    pickle.dump(regret_means, f)
            else:
                with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type 
                          + "/regrets_" + str(num_training_bids) + "_" + str(num_training_rounds) + ".pkl", "wb") as f:
                    pickle.dump(regret_means, f)
        print(f"Done with round {key}")
    


def boxplot_regrets(dist_type: str, zoom = True):
    # load regrets
    if dist_type == "real":
        with open("/u/scratch/y/yqg36/ADPP/data/real/regrets.pkl", "rb") as f:
            regrets = pickle.load(f)
        with open("/u/scratch/y/yqg36/ADPP/data/real/regrets_100_20.pkl", "rb") as f:
            RSRDE_regrets_100_20 = list(pickle.load(f).values())
#         with open("/u/scratch/y/yqg36/ADPP/data/real/regrets_100_200.pkl", "rb") as f:
#             RSRDE_regrets_100_200 = pickle.load(f)
        with open("/u/scratch/y/yqg36/ADPP/data/real/regrets_1000_20.pkl", "rb") as f:
            RSRDE_regrets_1000_20 = list(pickle.load(f).values())
#         with open("/u/scratch/y/yqg36/ADPP/data/real/regrets_1000_200.pkl", "rb") as f:
#             RSRDE_regrets_1000_200 = pickle.load(f) 
    else:
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets.pkl", "rb") as f:
            regrets = pickle.load(f)
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets_100_20.pkl", "rb") as f:
            RSRDE_regrets_100_20 = list(pickle.load(f).values())
#         with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets_100_200.pkl", "rb") as f:
#             RSRDE_regrets_100_200 = pickle.load(f)
        with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets_1000_20.pkl", "rb") as f:
            RSRDE_regrets_1000_20 = list(pickle.load(f).values())
#         with open("/u/scratch/y/yqg36/ADPP/data/sim/" + dist_type + "/regrets_1000_200.pkl", "rb") as f:
#             RSRDE_regrets_1000_200 = pickle.load(f) 
    DOP_regrets = regrets[0][:100] 
    RSOP_regrets = regrets[1][:100]
    RSKDE_regrets = regrets[2][:100]

    # generate figures
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax1 = fig.add_subplot()
    sns.boxplot([DOP_regrets, RSOP_regrets, RSKDE_regrets, 
                 RSRDE_regrets_100_20, RSRDE_regrets_1000_20],
                palette="Set2")
    plt.xticks([0, 1, 2, 3, 4], 
               ['DOP', 'RSOP', 'RSKDE', 'RSRDE\n100-20', 'RSRDE\n1000-20'])
    if zoom:
        ax2 = fig.add_axes([0.5, 0.5, 0.395, 0.37])
        sns.boxplot([RSRDE_regrets_100_20, RSRDE_regrets_1000_20],
                    palette=sns.color_palette("Set2")[3:5])
        plt.xticks([0, 1], 
                   ['RSRDE\n100-20', 'RSRDE\n1000-20'])

    # save and show figures
    plt.savefig(dist_type + ".png", bbox_inches='tight')
    plt.show()