import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from auth import register_user, login_user

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Menghapus kolom 'created_at' jika ada dalam dataset
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    
    return df

# Fungsi untuk melatih model dan mendeteksi anomali
def detect_anomalies(df, TDS_upper_limit, TDS_lower_limit):
    # Membagi data menjadi set pelatihan dan pengujian, menghapus baris yang memiliki missing values
    X_train, X_test = train_test_split(df.dropna(), test_size=0.2, random_state=42)

    # Mengimputasi nilai yang hilang dengan rata-rata
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Pelatihan model Isolation Forest dengan parameter yang disesuaikan
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train_imputed)

    # Prediksi anomali pada data pelatihan dan pengujian
    train_anomalies = model.predict(X_train_imputed)
    test_anomalies = model.predict(X_test_imputed)

    # Mengonversi hasil prediksi ke boolean (True untuk anomali, False untuk data normal)
    train_anomalies = train_anomalies == -1
    test_anomalies = test_anomalies == -1

    # Menambahkan label anomali ke DataFrame
    df_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    df_train['anomaly'] = train_anomalies
    df_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    df_test['anomaly'] = test_anomalies

    # Menambahkan logika berbasis aturan untuk mendeteksi TDS di luar batas sebagai anomali
    df_train['anomaly'] = df_train.apply(lambda row: True if (row['TDS'] > TDS_upper_limit or row['TDS'] < TDS_lower_limit) else row['anomaly'], axis=1)
    df_test['anomaly'] = df_test.apply(lambda row: True if (row['TDS'] > TDS_upper_limit or row['TDS'] < TDS_lower_limit) else row['anomaly'], axis=1)
    
    return df_train, df_test

# Streamlit App
st.title("Deteksi Anomali pada Data Air Nutrisi Hidroponik")

# Inisialisasi state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Fungsi untuk menampilkan halaman utama
def main_page():
    st.write(f"Selamat datang {st.session_state.username}")

    # Input file uploader
    uploaded_file = st.file_uploader("Unggah file dataset CSV", type="csv", key="file_uploader_main")

    # Parameter TDS
    TDS_upper_limit = st.number_input("Batas atas TDS:", value=1200, key="upper_limit")
    TDS_lower_limit = st.number_input("Batas bawah TDS:", value=400, key="lower_limit")

    if uploaded_file is not None:
        # Memuat dan memproses data
        df = load_and_preprocess_data(uploaded_file)
        
        if 'TDS' not in df.columns:
            st.error("Dataset harus memiliki kolom 'TDS'.")
        else:
            # Menampilkan data awal
            st.write("Data Awal:")
            st.dataframe(df)  # Menampilkan seluruh data yang diunggah
            
            # Deteksi anomali
            df_train, df_test = detect_anomalies(df, TDS_upper_limit, TDS_lower_limit)

            # Membuat kolom untuk menampilkan tabel secara berdampingan
            col1, col2 = st.columns(2)

            with col1:
                st.write("Data Pelatihan:")
                st.dataframe(df_train)

            with col2:
                st.write("Data Pengujian:")
                st.dataframe(df_test)

            # Visualisasi anomali
            st.write("Visualisasi Anomali:")
            fig, ax = plt.subplots()
            ax.plot(df_train.index, df_train['TDS'], label='TDS')
            anomalies_train = df_train[df_train['anomaly']]
            ax.scatter(anomalies_train.index, anomalies_train['TDS'], color='red', label='Anomali (Train)')
            anomalies_test = df_test[df_test['anomaly']]
            ax.scatter(anomalies_test.index, anomalies_test['TDS'], color='orange', label='Anomali (Test)')
            ax.legend()
            st.pyplot(fig)

# Fitur Login dan Register
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    st.subheader("Login")

    username = st.text_input("Username", key="username_login")
    password = st.text_input("Password", type='password', key="password_login")
    if st.button("Login", key="login_button"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Selamat datang {username}")
        else:
            st.warning("Username atau password salah")

elif choice == "Register":
    st.subheader("Buat Akun Baru")

    username = st.text_input("Username", key="username_register")
    password = st.text_input("Password", type='password', key="password_register")
    if st.button("Register", key="register_button"):
        if register_user(username, password):
            st.success("Akun berhasil dibuat")
            st.info("Silakan login dengan akun Anda")
        else:
            st.warning("Username sudah terdaftar")

# Menampilkan halaman utama jika pengguna sudah login
if st.session_state.logged_in:
    main_page()
