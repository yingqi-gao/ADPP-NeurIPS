import pandas as pd
import numpy as np
from _util import pickle_dump, REAL_DATA_PKL_PATH



def preprocess_real():
    df = pd.read_csv("./assets/auction97.csv")
    useful_df = df[["round", "item_name", "block", "bidder_name", "bid_amount"]].dropna()
    clean_df = useful_df.sort_values("round", ascending = False).groupby(["item_name", "bidder_name"]).first().reset_index()
    training_data = np.log(clean_df["bid_amount"][clean_df["block"] == "G"])
    pickle_dump(training_data, REAL_DATA_PKL_PATH)



def main():
    preprocess_real()


if __name__ == "__main__":
    main()