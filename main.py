import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
from auth import register_user, login_user

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Menghapus kolom 'created_at' jika ada dalam dataset
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    
    return df

# Fungsi untuk mendeteksi anomali
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

    # Menghitung metrik evaluasi untuk pelatihan dan pengujian
    y_true_train = df_train['anomaly']
    y_pred_train = model.predict(X_train_imputed) == -1

    y_true_test = df_test['anomaly']
    y_pred_test = model.predict(X_test_imputed) == -1

    # Evaluasi metrik untuk pengujian
    accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    precision = precision_score(y_true_test, y_pred_test) * 100
    recall = recall_score(y_true_test, y_pred_test) * 100
    f1 = f1_score(y_true_test, y_pred_test) * 100

    # Mengembalikan hasil evaluasi dan DataFrame
    return df_train, df_test, model, X_test_imputed, accuracy, precision, recall, f1, y_true_test, y_pred_test

# Fungsi untuk menampilkan halaman utama
def main_page():
    st.write(f"Selamat datang {st.session_state.username}")

    # Input file uploader
    uploaded_file = st.file_uploader("Unggah file dataset CSV", type="csv", key="file_uploader_main")

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
            df_train, df_test, model, X_test_imputed, accuracy, precision, recall, f1, y_true_test, y_pred_test = detect_anomalies(df, TDS_upper_limit, TDS_lower_limit)

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

            # Scatter Plot train
            st.write("Scatter Plot Data Train")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_train[df_train['anomaly'] == False]['pH'], df_train[df_train['anomaly'] == False]['TDS'], color='blue', label='Normal')
            ax.scatter(df_train[df_train['anomaly'] == True]['pH'], df_train[df_train['anomaly'] == True]['TDS'], color='red', label='Anomaly')
            ax.set_xlabel('pH')
            ax.set_ylabel('TDS')
            ax.set_title('Hasil Pelatihan Scatter Plot pH dan TDS dengan Deteksi Anomali')
            ax.legend()
            st.pyplot(fig)


            # Scatter Plot uji
            st.write("Scatter Plot Data Uji")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_test[df_test['anomaly'] == False]['pH'], df_test[df_test['anomaly'] == False]['TDS'], color='blue', label='Normal')
            ax.scatter(df_test[df_test['anomaly'] == True]['pH'], df_test[df_test['anomaly'] == True]['TDS'], color='red', label='Anomaly')
            ax.set_xlabel('pH')
            ax.set_ylabel('TDS')
            ax.set_title('Hasil Uji Scatter Plot pH dan TDS dengan Deteksi Anomali')
            ax.legend()
            st.pyplot(fig)

            # Menghitung dan menampilkan metrik evaluasi
            st.write(f"Akurasi: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"F1 Score: {f1:.2f}%")

            # Data metrik
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [accuracy, precision, recall, f1]  # Nilai metrik
            colors = ['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB']  # Warna berbeda untuk setiap metrik
            
            # Membuat histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metrics, values, color=colors)
            
            # Menambahkan detail pada histogram
            ax.set_title('Histogram Metrik Evaluasi', fontsize=16, color='#D3D3D3')
            ax.set_ylabel('Persentase (%)', fontsize=12, color='#D3D3D3')
            ax.set_ylim(0, 100)  # Karena metrik dalam skala persentase
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=10, color='#D3D3D3')
            ax.yaxis.set_tick_params(colors='#D3D3D3')
            
            # Menambahkan nilai pada atas setiap batang histogram
            for i, v in enumerate(values):
                ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10, color='#FFFFFF')
            
            # Menyesuaikan latar belakang untuk dark mode
            fig.patch.set_facecolor('#2F2F2F')  # Latar luar
            ax.set_facecolor('#2F2F2F')  # Latar dalam
            
            # Tampilkan plot di Streamlit
            st.pyplot(fig)

            # Menampilkan laporan evaluasi
            st.write("Laporan Evaluasi (Data Pengujian):")
            report = classification_report(y_true_test, y_pred_test, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

# Fungsi login
def login():
    st.title("Website Deteksi Anomali Pada Nutrisi Air Hidroponik")
    st.subheader("Login")

    username = st.text_input("Username", key="username_login")
    password = st.text_input("Password", type='password', key="password_login")
    if st.button("Login", key="login_button"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Selamat datang {username}")
            # Set session state for rerun and prevent continuous reruns
            if 'rerun' not in st.session_state:
                st.session_state.rerun = True
                st.experimental_rerun()
        else:
            st.warning("Username atau password salah")

# Fungsi register
def register():
    st.title("Website Deteksi Anomali Pada Nutrisi Air Hidroponik")
    st.subheader("Buat Akun Baru")

    username = st.text_input("Username", key="username_register")
    password = st.text_input("Password", type='password', key="password_register")
    if st.button("Register", key="register_button"):
        if register_user(username, password):
            st.success("Akun berhasil dibuat")
            st.info("Silakan login dengan akun Anda")
        else:
            st.warning("Username sudah terdaftar")

# Inisialisasi state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Tampilkan menu login atau register hanya jika belum login
menu = ["Login", "Register", "Main Page"]
if st.session_state.logged_in:
    menu.remove("Login")
    menu.remove("Register")
else:
    menu.remove("Main Page")

choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    login()
elif choice == "Register":
    register()
elif choice == "Main Page":
    main_page()
