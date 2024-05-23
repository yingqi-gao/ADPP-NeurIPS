import os
import dill

REAL_DATA_PKL_PATH = "./assets/real_data.pkl"
TEST_REPETITIONS = 100
PARTITION_REPETITIONS = 200

def get_regrets_uri(*, output_uri: str, dist_type: str, mechanism: str, num_training_bids: int | None = None, num_training_rounds: int | None = None):
    regrets_output_dir_uri = os.path.join(output_uri, dist_type)
    if mechanism == "RSRDE":
        return os.path.join(regrets_output_dir_uri, f"regrets_{'_'.join(['RSRDE', str(num_training_bids), str(num_training_rounds)])}.pkl")
    return os.path.join(regrets_output_dir_uri, f"regrets_{mechanism}.pkl")


def pickle_dump(object, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        dill.dump(object, f)


def pickle_load(file_path: str):
    with open(file_path, "rb") as f:
        return dill.load(f)
