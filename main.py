import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Menghapus kolom 'created_at' jika ada dalam dataset
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    
    return df

# Fungsi untuk mendeteksi anomali menggunakan K-Fold Cross-Validation
def detect_anomalies_with_cv(df, TDS_upper_limit, TDS_lower_limit, n_splits=5):
    # Tambahkan kolom anomali_ground_truth
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")
    
    df['anomali_ground_truth'] = False
    df.loc[((df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit)), 'anomali_ground_truth'] = True
    
    # Memisahkan fitur dan target
    X = df.drop(columns=['anomali_ground_truth'])
    y = df['anomali_ground_truth']

    # Inisialisasi model
    model = IsolationForest(contamination=0.1, random_state=42)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Imputasi nilai hilang
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Melatih model
        model.fit(X_train_imputed)

        # Prediksi
        y_pred = model.predict(X_test_imputed) == -1  # Menandakan anomali sebagai -1

        # Menghitung metrik evaluasi
        accuracies.append(accuracy_score(y_test, y_pred) * 100)
        precisions.append(precision_score(y_test, y_pred) * 100)
        recalls.append(recall_score(y_test, y_pred) * 100)
        f1_scores.append(f1_score(y_test, y_pred) * 100)

    # Menghitung rata-rata metrik
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1, model

# Fungsi untuk menampilkan halaman utama
def main_page():
    st.write("Selamat datang di Aplikasi Deteksi Anomali pada Nutrisi Air Hidroponik")

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

            # Deteksi anomali dengan K-Fold Cross-Validation
            try:
                avg_accuracy, avg_precision, avg_recall, avg_f1, model = detect_anomalies_with_cv(df, TDS_upper_limit, TDS_lower_limit)
            except ValueError as e:
                st.error(str(e))
                return

            # Menampilkan metrik evaluasi rata-rata
            st.write(f"Akurasi Rata-rata: {avg_accuracy:.2f}%")
            st.write(f"Precision Rata-rata: {avg_precision:.2f}%")
            st.write(f"Recall Rata-rata: {avg_recall:.2f}%")
            st.write(f"F1 Score Rata-rata: {avg_f1:.2f}%")

            # Data metrik
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [avg_accuracy, avg_precision, avg_recall, avg_f1]
            colors = ['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB']
            
            # Membuat histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metrics, values, color=colors)
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

            # Laporan evaluasi
            st.write("Laporan Evaluasi: ")
            st.write(pd.DataFrame(classification_report(df['anomali_ground_truth'], model.predict(df.drop(columns=['anomali_ground_truth'])) == -1, output_dict=True)).transpose())

# Menjalankan halaman utama
main_page()
