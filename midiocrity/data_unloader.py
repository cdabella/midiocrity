import pickle
import os


def unload_data(filename='../data/batches/tensors-0.pkl'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(str(dir_path))
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    unload_data()
