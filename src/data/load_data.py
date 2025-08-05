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

    features = data[['Submitted via', 'State', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company public response',
              'Timely response?', 'Days between']]
    targets = data['Company response to consumer']

    categorical_cols = ['Submitted via', 'State', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
                        'Company public response', 'Timely response?']
    numeric_cols = ['Days between']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num' , StandardScaler(),numeric_cols),
            ('cat' , OneHotEncoder(drop='first' , sparse_output =False))
        ]
    )
    X = data[features]
    y = data[targets]

    X_processed = preprocessor.fit_transform(X)
    X_train , X_test , y_train , y_test = train_test_split(X_processed , y , test_size = test_size,random_state=random_state)


