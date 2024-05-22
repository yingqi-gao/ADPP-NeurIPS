from _util import boxplot_regrets
import sys



def plot(dist_type: str):
    boxplot_regrets(dist_type, False)



def main():
    dist_type = sys.argv[1]
    plot(dist_type)



if __name__ == "__main__":
    main()