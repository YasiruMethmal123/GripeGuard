import pandas as pd
from fontTools.mtiLib import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def preprocess_text_data(X_train, X_test, vocab_size, max_length):
    """
    Tokenizes and pads text data. This would be in a file like
    `src/utils/data_preprocessor.py`.
    """
    trunc_type = 'post'
    padding_type = 'post'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
    tokenizer.fit_on_texts(X_train)

    training_sequences = tokenizer.text_to_sequence(X_train)
    testing_sequence = tokenizer.text_to_sequence(X_test)

    padded_training_sequence = pad_sequences(
        training_sequences,maxlen=max_length,padding=padding_type,truncating= trunc_type
    )
    padded_testing_sequence = pad_sequences(
        testing_sequence,maxlen=max_length,padding=padding_type,truncating=trunc_type
    )

    print("Text preprocessing complete.")
    return padded_training_sequence,padded_testing_sequence,tokenizer


