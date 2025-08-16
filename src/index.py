import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.process_text import preprocess_text_data
from src.model.evaluate_model import train_and_evaluate
from src.model.model import build_model
from src.model.prediction import predict_new_complaint

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 50
NUM_EPOCHS = 10

# Main execution flow
X_train, X_test, y_train, y_test, label_encoder = load_data("../resources/ConsumerComplaints.csv")

padded_X_train, padded_X_test, tokenizer = preprocess_text_data(
    X_train, X_test, VOCAB_SIZE, MAX_LENGTH
)

num_classes = len(label_encoder.classes_)
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, num_classes)

train_and_evaluate(
    model, padded_X_train, y_train, padded_X_test, y_test, NUM_EPOCHS, label_encoder
)

predict_new_complaint(model, tokenizer, label_encoder, MAX_LENGTH)