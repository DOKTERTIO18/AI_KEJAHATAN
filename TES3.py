import streamlit as st
import pickle
import numpy as np

# Memuat kembali model dan scaler dari file
loaded_model, loaded_scaler = pickle.load(open('model_kejahatan.sav', 'rb'))

# Judul web
st.title('PREDIKTOR TINGKAT KEJAHATAN')

# Membagi kolom
col1, col2 = st.columns(2)

with col1:
    Nyawa = st.text_input('Input Jumlah Nyawa')

with col2:
    FisikBadan = st.text_input('Input Jumlah Fisik/Badan')

with col1:
    Kesusilaan = st.text_input('Input Jumlah Kesusilaan')

with col2:
    KemerdekaanOrang = st.text_input('Input Jumlah Kemerdekaan Orang')

with col1:
    HakMilikBarangdenganPenggunaanKekerasan = st.text_input('Input Jumlah Hak Milik/Barang dengan Penggunaan Kekerasan')

with col2:
    HakMilikBarang = st.text_input('Input Jumlah Hak Milik/Barang')

with col1:
    Narkotika = st.text_input('Input Jumlah Narkotika')

with col2:
    PenipuanPenggelapandanKorupsi = st.text_input('Input Jumlah Penipuan, Penggelapan, dan Korupsi')

# Code untuk prediksi
kejahatan_highandlow = ''

# Membuat tombol untuk prediksi
if st.button('Test Prediksi Kejahatan'):
    # Memeriksa apakah input valid
    if Nyawa and FisikBadan and Kesusilaan and KemerdekaanOrang and HakMilikBarangdenganPenggunaanKekerasan and HakMilikBarang and Narkotika and PenipuanPenggelapandanKorupsi:
        # Mengonversi input ke float
        input_data = np.array([float(Nyawa), float(FisikBadan), float(Kesusilaan), float(KemerdekaanOrang),
                               float(HakMilikBarangdenganPenggunaanKekerasan), float(HakMilikBarang),
                               float(Narkotika), float(PenipuanPenggelapandanKorupsi)])

        # Menyesuaikan dimensi input untuk sesuaikan dengan scaler yang dimuat
        input_data_reshape = input_data.reshape(1, -1)

        # Melakukan prediksi dengan model yang dimuat kembali (loaded_model)
        std_input_data = loaded_scaler.transform(input_data_reshape)
        kejahatan_prediction = loaded_model.predict(std_input_data)

        # Menentukan tingkat kejahatan berdasarkan hasil prediksi
        if kejahatan_prediction[0] == 0:
            kejahatan_highandlow = 'Tingkat Kejahatan Rendah'
        else:
            kejahatan_highandlow = 'Tingkat Kejahatan Tinggi'

        st.success(kejahatan_highandlow)
    else:
        st.warning('Silakan masukkan nilai untuk semua kolom.')
