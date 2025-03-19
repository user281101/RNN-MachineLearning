import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#import keras 
#from keras import Sequential,layers
from tensorflow.keras import Sequential,models
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense

import matplotlib
matplotlib.use('Agg') 

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
    tf.keras.layers.LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(100, activation='tanh',return_sequences=True),
    tf.keras.layers.LSTM(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Trainieren des Modells
history = model.fit(X_train, y_train, validation_split=0.2, epochs=1, batch_size=64)

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

#schau warum die syntax nicht funktioniert dann sollte der code funktionieren

'''import tensorflow as tf
print(tf.__version__)  # Sollte 2.16.2 ausgeben

from tensorflow.keras import layers
print(layers.Dense(10))  # Testet, ob Keras funktioniert'''
