import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential,models
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('Agg') 

# Prüfen, ob Metal GPU verfügbar ist
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"GPU verfügbar: {gpu_devices[0]}")
else:
    print("Keine GPU gefunden.")

# Prüfen, welches Gerät für TensorFlow-Operationen genutzt wird
print("TensorFlow nutzt aktuell:", tf.test.gpu_device_name())


# Step 1: Laden und Vorbereiten von Daten
#data = pd.read_csv('jena_climate_2009_2016.csv')
data = pd.read_csv('/Users/knutvietthangfranke/Desktop/RNN-MachineLearning/jena_climate_2009_2016.csv')


# 'Datum/Uhrzeit' in datetime umwandeln und als Index festlegen
data['Date Time'] = pd.to_datetime(data['Date Time'], dayfirst=True)
data.set_index('Date Time', inplace=True)

#Daten auf stündliche Durchschnittswerte umrechnen
data = data.resample('H').mean()

plt.figure(figsize=(10, 6))
plt.plot(data['T (degC)'], label='Temperature (°C)')
plt.title('Temperature over Time')
plt.legend()
plt.savefig('temperature_over_time.png')  

# Daten in Trainings- und Testsätze aufteilen
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Skalieren von Daten mit MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Step 2: Sequenzen erstellen
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length, :-1])
        labels.append(data[i + sequence_length, -1])
    return np.array(sequences), np.array(labels)

sequence_length = 24  # Temperaturvorhersage basierend auf den letzten 24 Stunden
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

# Step 3: Erstellen des LSTM-Modells
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.BatchNormalization(),  # Stabilisiert das Training
    tf.keras.layers.LSTM(50, activation='tanh'),
    tf.keras.layers.Dropout(0.2),  # Verhindert Overfitting
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
#model.compile(optimizer='adam', loss='mse')

# Early Stopping Callback hinzufügen
early_stopping = EarlyStopping( 
    monitor='val_loss',  # Überwacht den Validierungsverlust
    patience=3,          # Stoppt, wenn sich val_loss für 3 Epochen nicht verbessert
    restore_best_weights=True  # Nimmt die besten Modellgewichte
)

# Trainieren des Modells
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)
#callbacks=[early_stopping]

# Step 4: Bewerten das Modell
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")

# Step 5: Visualize Predictions
predictions = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Temperatures')
plt.plot(predictions, label='Predicted Temperatures')
plt.title('Actual vs Predicted Temperatures')
plt.legend()
plt.savefig('actual_vs_predicted_temperatures.png')  

# Step 6: Trainingsverlauf plotten
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('loss_over_epochs.png')  

#gegenüberstellung
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual Temperatures', color='blue')
plt.plot(predictions[:100], label='Predicted Temperatures', linestyle='dashed', color='red')
plt.title('Actual vs Predicted Temperatures (Ausschnitt)')
plt.legend()
plt.savefig('actual_vs_predicted_temperatures.png')

import numpy as np

print(np.isnan(X_train).sum())  # Prüfen, ob NaNs im Trainingsdatensatz sind
print(np.isnan(y_train).sum())  # Prüfen, ob NaNs in den Labels sind

#accuracy score mit einfügen, schauen was der graph mir sagt
#mse metrik für regression ist ja temp-vorhersage
#morgen videos zu rnn und lstm schauen
#python tutorial von codebasics durchgehen, neben ML auch python skills aufbauen und regelmäßig ins CV schreiben und bewerben