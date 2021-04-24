import pickle
import os


def unload_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(str(dir_path))
    with open('tensors-0.pkl', 'rb') as f:
        data = pickle.load(f)
        print("hoot")


if __name__ == "__main__":
    unload_data()
