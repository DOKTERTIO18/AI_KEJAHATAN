import numpy as np
import pickle

# Fungsi untuk menghitung prediksi berdasarkan input data
def predict_crime(input_data):
    # Memuat kembali model dan scaler dari file
    filename = 'model_kejahatan.sav'
    loaded_model, loaded_scaler = pickle.load(open(filename, 'rb'))

    # Memastikan input_data memiliki jumlah fitur yang sesuai
    expected_features = 8  # Sesuaikan dengan jumlah fitur yang diharapkan oleh scaler saat melatih model
    if len(input_data) != expected_features:
        raise ValueError(f"Input data has {len(input_data)} features, but model expects {expected_features} features.")

    # Standarisasi data input menggunakan scaler yang dimuat kembali
    std_input_data = loaded_scaler.transform([input_data])

    # Melakukan prediksi dengan model yang dimuat kembali (loaded_model)
    prediction = loaded_model.predict(std_input_data)

    # Mengembalikan hasil prediksi
    return prediction[0]

# Contoh penggunaan model untuk data input baru
input_data = [15368, 251468, 245896, 199025, 210254, 187895, 125486, 203654]

# Memanggil fungsi predict_crime untuk melakukan prediksi
try:
    prediction = predict_crime(input_data)
    if prediction == 0:
        print('Kejahatan tingkat tinggi')
    else:
        print('Kejahatan tingkat rendah')

except ValueError as ve:
    print(str(ve))
