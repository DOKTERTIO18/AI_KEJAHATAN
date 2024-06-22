import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
import pickle

# Membaca dataset
try:
    kejahatan_data = pd.read_excel('dataset kejahatan.xlsx')
except FileNotFoundError:
    print("File 'dataset kejahatan.xlsx' tidak ditemukan. Pastikan nama file dan lokasinya sesuai.")

# Memisahkan fitur (X) dan target (Y)
X = kejahatan_data.drop(columns='outcome', axis=1)
Y = kejahatan_data['outcome']

# Standarisasi data
scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)

# Memisahkan data menjadi data training dan data testing
X_train, X_test, Y_train, Y_test = train_test_split(standarized_data, Y, test_size=0.2, stratify=Y, random_state=9)

# Membuat model SVM dengan kernel linear
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluasi akurasi data training
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Akurasi data training:', training_data_accuracy)

# Evaluasi akurasi data testing
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Akurasi data testing:', test_data_accuracy)

# Menyimpan model dan scaler ke dalam file
filename = 'model_kejahatan.sav'
pickle.dump((classifier, scaler), open(filename, 'wb'))
