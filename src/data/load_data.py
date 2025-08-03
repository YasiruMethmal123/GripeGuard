from errno import ELOOP

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

from tensorboard.compat.tensorflow_stub.io.gfile import exists


def load_and_preprocess_data(test_size = 0.2 , random_state = 42):

    file_path = "../resources/ConsumerComplaints.csv"  # Adjust to match exact filename
    print("Current working directory:", os.getcwd())
    abs_path = os.path.abspath(file_path)
    print("Looking for file at:", abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found at {abs_path}. Please ensure the file exists.")
    data = pd.read_csv(abs_path)
    print("File loaded successfully. First few rows:")
    print(data.head())

    

    return data


load_and_preprocess_data()