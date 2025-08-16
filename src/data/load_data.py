

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import os

from tensorboard.compat.tensorflow_stub.io.gfile import exists


def load_data(test_size = 0.2 , random_state = 42 ,update_original=False):
    """
    Loads and preprocesses the data. In a real project, this function
    would likely be in a separate file like `src/utils/data_loader.py`.
    """

    file_path = "../resources/ConsumerComplaints.csv"  # Adjust to match exact filename
    print("Current working directory:", os.getcwd())
    abs_path = os.path.abspath(file_path)
    print("Looking for file at:", abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found at {abs_path}. Please ensure the file exists.")
    df = pd.read_csv(abs_path)
    print("File loaded successfully. First few rows:")
    print(df.head())

    # Combine text features and drop missing values
    df['Issue'].fillna('', inplace=True)
    df['Sub-issue'].fillna('', inplace=True)
    df['text'] = df['Issue'] + ' ' + df['Sub-issue']
    df.dropna(subset=['Product'], inplace=True)

    X = df['text']
    y = df['Product']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Data loaded and split.")
    return X_train, X_test, y_train, y_test, label_encoder





