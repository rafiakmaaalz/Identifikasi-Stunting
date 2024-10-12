import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the trained model and scaler
model_stunting = pickle.load(open('deployment.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Set the page title and icon
st.set_page_config(page_title="Identifikasi Stunting", page_icon="🏥")

# Title and description
st.title('Website Identifikasi Stunting')
st.markdown("""
Masukkan nilai-nilai di bawah ini untuk mengidentifikasi stunting pada anak.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input fields in the first column
with col1:
    Usia = st.text_input('Input Nilai Usia (bulan)', placeholder="Contoh: 24")
    Berat = st.text_input('Input Nilai Berat (kg)', placeholder="Contoh: 12.5")
    Tinggi = st.text_input('Input Nilai Tinggi (cm)', placeholder="Contoh: 90")

# Input fields in the second column
with col2:
    BBU = st.text_input('Input Nilai BB/U', placeholder="Contoh: 0/1/2/3/4")
    BBTB = st.text_input('Input Nilai BB/TB', placeholder="Contoh: 0/1/2/3/4")
    Gender = st.selectbox('Input Nilai Gender (Laki laki = 0, Perempuan = 1)', ('Laki-laki', 'Perempuan'))

# Add a separator for better visual organization
st.markdown("---")

identifikasi = ''

# Button for prediction
if st.button('Identifikasi Stunting'):
    try:
        # Validate if inputs are not empty
        if not Usia or not Berat or not Tinggi or not BBU or not BBTB:
            st.error('Semua input harus diisi.')
        else:
            # Convert inputs to the appropriate types
            usia_val = float(Usia)
            berat_val = float(Berat)
            tinggi_val = float(Tinggi)
            bbu_val = float(BBU)
            bbtb_val = float(BBTB)

            # Convert gender input to a numeric value
            gender_val = 0 if Gender == 'Laki-laki' else 1

            # Create input array for prediction
            input_data = np.array([[usia_val, berat_val, tinggi_val, bbu_val, bbtb_val, gender_val]])

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Debug: Print input values
            st.write("Input data for prediction:", input_data_scaled)

            # Make prediction
            stunting_pred = model_stunting.predict(input_data_scaled)

            # Debug: Print prediction
            st.write("Model prediction:", stunting_pred)

            # Interpret the prediction
            if stunting_pred[0] == 0:  # Assuming 0 = 'Normal'
                identifikasi = 'Tidak terkena stunting (Normal)'
            elif stunting_pred[0] == 1:  # Assuming 1 = 'Pendek'
                identifikasi = 'Terkena stunting (Pendek)'
            else:  # Assuming 2 = 'Sangat Pendek'
                identifikasi = 'Terkena stunting (Sangat Pendek)'

            # Display result with emphasis
            st.success(identifikasi)

    except ValueError as e:
        st.error(f'Input error: {e}')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')

# Add some footer or additional information
st.markdown("---")
st.markdown("""
**Catatan:** Semua input harus berupa angka positif. Pastikan untuk memeriksa kembali nilai yang dimasukkan.
""")

# Upload CSV file
st.markdown("---")
st.subheader("Data dari CSV Yang Digunakan Untuk Indentifikasi")

# Load and display CSV data
csv_data = 'NewStunting.csv'  # Ganti dengan path ke file CSV Anda
try:
    data = pd.read_csv(csv_data)
    st.dataframe(data)  # Menampilkan data sebagai tabel
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan path sudah benar.")
except Exception as e:
    st.error(f'Kesalahan saat memuat data: {e}')







