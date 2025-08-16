from tabnanny import verbose

from keras.src.utils import pad_sequences
import tensorflow as tf

def predict_new_complaint(model, tokenizer, label_encoder, max_length):
    """
    Demonstrates a prediction on new text. This function could be in a
    separate script for inference, e.g., `src/predict.py`.
    """
    print("\n--- Example Prediction ---")

    new_compliant = "Incorrect information on my credit report is causing issues."
    new_compliant_sequence = tokenizer.texts_to_sequances([new_compliant])
    padded_new_complaint = pad_sequences(
        new_compliant_sequence , maxlen=max_length,padding='post' , truncating= 'post'
    )

    prediction_probs = model.predict(padded_new_complaint , verbose = 0)
    prediction_class = tf.argmax(prediction_probs , axis= 1).numpy()[0]
    prediction_product = label_encoder.inverse_transform([prediction_class])[0]

    print(f"New complaint text:'{new_compliant}'")
    print(f"Predicted product:'{prediction_product}'")
