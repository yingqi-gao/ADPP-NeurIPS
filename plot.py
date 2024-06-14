from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import seaborn as sns

from _util import pickle_load


def plot(*, dist_type: str, output_uri: str, zoom=False):
    # load regrets
    regrets = {}
    dir_name = os.path.join(output_uri, dist_type)
    for filename in os.listdir(dir_name):
        if filename.startswith("regrets"):
            name = os.path.basename(filename).split(".")[0].lstrip("regrets_")
            data = pickle_load(os.path.join(dir_name, filename))
            if "RSRDE" in name:
                data = list(data.values())
            regrets[name] = data

    # generate figures
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax1 = fig.add_subplot()

    sns.boxplot(list(regrets.values()), palette="Set2")
    labels = [
        key.replace("_", "__", 1).replace("__", "\n", 1) for key in regrets.keys()
    ]
    plt.xticks(range(len(labels)), labels)
    if zoom:
        ax2 = fig.add_axes([0.5, 0.5, 0.395, 0.37])
        sns.boxplot(
            [
                regrets["RSRDE_1000_20"],
                regrets["RSRDE_1000_200"],
                regrets["RSRDE_100_20"],
                regrets["RSRDE_100_200"],
            ],
            palette=sns.color_palette("Set2")[3:7],
        )
        plt.xticks(
            [0, 1, 2, 3],
            ["RSRDE\n1000-20", "RSRDE\n1000-200", "RSRDE\n100-20", "RSRDE\n100-200"],
        )

    # save and show figures
    plt.savefig(os.path.join(output_uri, f"{dist_type}.png"), bbox_inches="tight")
    plt.show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dist-type",
        required=True,
        help="Specify the distribution family or 'real'.",
        choices=["uniform", "normal", "exponential", "real"],
    )
    parser.add_argument("--output-uri", default="./sim/")
    parser.add_argument(
        "--zoom", help="Should we zoom in for RSRDE regrets?", choices=["Yes", "No"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot(
        dist_type=args.dist_type, output_uri=args.output_uri, zoom=(args.zoom == "Yes")
    )


if __name__ == "__main__":
    main()
