from keras.src.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.python.keras.models import Sequential


def build_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    Builds the deep learning model. This function would be in a file like
    `src/models/model.py`.
    """

    print("\nBuilding deep learning model....")
    model = Sequential({
        Embedding(vocab_size , embedding_dim , input_length= max_length),
        GlobalAveragePooling1D()
        Dense(24 , activation='relu'),
        Dense(num_classes , activation='softmax')
    })

    model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam',matrics =['accuracy'])
    model.summary()
    return model