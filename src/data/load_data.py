import shutil
from errno import ELOOP

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

from tensorboard.compat.tensorflow_stub.io.gfile import exists


def load_and_preprocess_data(test_size = 0.2 , random_state = 42 ,update_original=False):

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
              'Timely response?']]
    targets = data['Company response to consumer']

    missing_cols = [col for col in features + [targets] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    categorical_cols = ['Submitted via', 'State', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
                        'Company public response', 'Timely response?']
    numeric_cols = ['Days between']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ] if categorical_cols else []
    )
    X = data[features]
    y = data[targets]

    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(cat_feature_names)

    # Convert processed features to DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=data.index)

    # Combine processed features with target
    processed_data = pd.concat([X_processed_df, y.rename('Company response to consumer')], axis=1)


    X_processed = preprocessor.fit_transform(X)
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        print("Data split successfully. Training set shape:", X_train.shape, "Test set shape:", X_test.shape)
    except Exception as e:
        print("Error in train_test_split:", str(e))
        raise

    if update_original:
        # Create a backup of the original file
        backup_path = abs_path + '.backup'
        shutil.copyfile(abs_path, backup_path)
        print(f"Backup of original file created at: {backup_path}")

        # Save processed data to the original CSV
        processed_data.to_csv(abs_path, index=False)
        print(f"Processed data saved to original file: {abs_path}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    try:
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
            test_size=0.2, random_state=42, update_original=False
        )
        print("Data preprocessing complete.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
    except Exception as e:
        print("Error in main execution:", str(e))
        raise