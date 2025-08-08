from numpy.f2py.crackfortran import verbose
import tensorflow as tf
from sklearn.metrics import classification_report

import model
def train_and_evaluate(model, padded_X_train, y_train, padded_X_test, y_test, num_epochs, label_encoder):
    """
    Trains and evaluates the model. This would be in a file like
    `src/training/trainer.py`.
    """
    print("\n Training the model")
    model.fit(
        padded_X_train,y_train, epochs=num_epochs,
        validation_data = (padded_X_test,y_test) , verbose = 2
    )

    print("\nEvaluating the model .....")
    loss , accuracy = model.evaluate(padded_X_test , y_test , verbose =0)
    print(f"Test Accuracy : {accuracy*100:.2f}%")

    y_pred_probs =  model.predict(padded_X_test , verbose = 0)
    y_pred = tf.argmax(y_pred_probs , axis=1).numpy()
    print("\nClassification Report:\n", classification_report(y_test , y_pred , target_names= label_encoder.classe_))