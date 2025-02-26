import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Menghapus kolom 'created_at' jika ada dalam dataset
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    
    return df

# Fungsi untuk mendeteksi anomali
def detect_anomalies(df, TDS_upper_limit, TDS_lower_limit):
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")

    df['anomali_ground_truth'] = False
    df.loc[((df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit)), 'anomali_ground_truth'] = True

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    if 'anomali_ground_truth' not in X_test.columns:
        raise ValueError("Kolom 'anomali_ground_truth' hilang setelah train_test_split.")

    # Imputasi nilai hilang
    if X_train.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train.drop(columns='anomali_ground_truth'))
        X_test_imputed = imputer.transform(X_test.drop(columns='anomali_ground_truth'))
    else:
        X_train_imputed = X_train.drop(columns='anomali_ground_truth').values
        X_test_imputed = X_test.drop(columns='anomali_ground_truth').values

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train_imputed)

    train_anomalies = model.predict(X_train_imputed) == -1
    test_anomalies = model.predict(X_test_imputed) == -1

    df_train = pd.DataFrame(X_train_imputed, columns=X_train.drop(columns='anomali_ground_truth').columns)
    df_train['anomali'] = train_anomalies

    df_test = pd.DataFrame(X_test_imputed, columns=X_test.drop(columns='anomali_ground_truth').columns)
    df_test['anomali'] = test_anomalies
    df_test['anomali_ground_truth'] = X_test['anomali_ground_truth'].values

    y_true_test = df_test['anomali_ground_truth']
    y_pred_test = df_test['anomali']
    accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    precision = precision_score(y_true_test, y_pred_test) * 100
    recall = recall_score(y_true_test, y_pred_test) * 100
    f1 = f1_score(y_true_test, y_pred_test) * 100

    return df_train, df_test, model, accuracy, precision, recall, f1

# Fungsi utama
def main_page():
    st.write("Selamat datang di Aplikasi Deteksi Anomali pada Nutrisi Air Hidroponik")

    uploaded_file = st.file_uploader("Unggah file dataset CSV", type="csv", key="file_uploader_main")

    TDS_upper_limit = st.number_input("Batas atas TDS:", value=1200, key="upper_limit")
    TDS_lower_limit = st.number_input("Batas bawah TDS:", value=400, key="lower_limit")
    
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        
        if 'TDS' not in df.columns:
            st.error("Dataset harus memiliki kolom 'TDS'.")
        else:
            st.write("Data Awal:")
            st.dataframe(df)

            try:
                df_train, df_test, model, accuracy, precision, recall, f1 = detect_anomalies(df, TDS_upper_limit, TDS_lower_limit)
            except ValueError as e:
                st.error(str(e))
                return

            col1, col2 = st.columns(2)

            with col1:
                st.write("Data Pelatihan:")
                st.dataframe(df_train)

            with col2:
                st.write("Data Pengujian:")
                st.dataframe(df_test)

            # Scatter Plot train
            st.write("Scatter Plot Data Train")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_train[df_train['anomali'] == False]['pH'], df_train[df_train['anomali'] == False]['TDS'], color='blue', label='Normal')
            ax.scatter(df_train[df_train['anomali'] == True]['pH'], df_train[df_train['anomali'] == True]['TDS'], color='red', label='Anomali')
            ax.set_xlabel('pH')
            ax.set_ylabel('TDS')
            ax.set_title('Hasil Pelatihan Scatter Plot pH dan TDS dengan Deteksi Anomali')
            ax.legend()
            ax.set_ylim(800, 1400)  # BATAS TDS DIPERBAIKI
            st.pyplot(fig)

            # Scatter Plot uji
            st.write("Scatter Plot Data Uji")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_test[df_test['anomali'] == False]['pH'], df_test[df_test['anomali'] == False]['TDS'], color='blue', label='Normal')
            ax.scatter(df_test[df_test['anomali'] == True]['pH'], df_test[df_test['anomali'] == True]['TDS'], color='red', label='Anomali')
            ax.set_xlabel('pH')
            ax.set_ylabel('TDS')
            ax.set_title('Hasil Uji Scatter Plot pH dan TDS dengan Deteksi Anomali')
            ax.legend()
            ax.set_ylim(800, 1400)  # BATAS TDS DIPERBAIKI
            st.pyplot(fig)

            # Menampilkan metrik evaluasi
            st.write(f"Akurasi: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"F1 Score: {f1:.2f}%")

            # Histogram Metrik
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [accuracy, precision, recall, f1]
            colors = ['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metrics, values, color=colors)
            ax.set_title('Histogram Metrik Evaluasi', fontsize=16, color='#D3D3D3')
            ax.set_ylabel('Persentase (%)', fontsize=12, color='#D3D3D3')
            ax.set_ylim(0, 100)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=10, color='#D3D3D3')
            ax.yaxis.set_tick_params(colors='#D3D3D3')

            for i, v in enumerate(values):
                ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10, color='#FFFFFF')

            fig.patch.set_facecolor('#2F2F2F')
            ax.set_facecolor('#2F2F2F')
            
            st.pyplot(fig)

            # Laporan Evaluasi
            st.write("Laporan Evaluasi:")
            st.write(pd.DataFrame(classification_report(df_test['anomali_ground_truth'], df_test['anomali'], output_dict=True)).transpose())

# Menjalankan halaman utama
main_page()
