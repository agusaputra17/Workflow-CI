import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import warnings
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

df = pd.read_csv('banking-data_preprocessing.csv')
X = df.drop(columns=['y'])
y = df['y']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Bangun Model Neural Network
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Output 0 atau 1
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("mlflow-keras-banking")

with mlflow.start_run():

    # Log model parameters
    mlflow.tensorflow.autolog()

    # Fit model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluasi
    score = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation loss: {score[0]:.4f}")
    print(f"Validation accuracy: {score[1]:.2f}")

    # Prediksi
    y_pred = (model.predict(X_val) > 0.5).astype("int32").flatten()

    # Hitung metrik
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Buat confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Plot dan simpan confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Simpan gambar
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log metrik
    mlflow.log_metric("accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("loss", history.history['loss'][-1])
    mlflow.log_metric("val_loss", score[0])
    mlflow.log_metric("val_accuracy", score[1])
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)
    mlflow.log_metric("val_f1_score", f1)
    mlflow.log_artifact("confusion_matrix.png")

    # Print hasil
    print(f"Validation precision: {precision:.2f}")
    print(f"Validation recall: {recall:.2f}")
    print(f"Validation F1 Score: {f1:.2f}")
