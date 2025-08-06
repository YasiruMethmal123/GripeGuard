import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def preprocess_text_data(X_train, X_test, vocab_size, max_length):
    """
    Tokenizes and pads text data. This would be in a file like
    `src/utils/data_preprocessor.py`.
    """