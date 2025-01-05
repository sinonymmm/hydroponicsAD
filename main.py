import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Menghapus kolom 'created_at' jika ada dalam dataset
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    
    return df

# Fungsi untuk deteksi anomali dengan cross-validation
def detect_anomalies(df, TDS_upper_limit, TDS_lower_limit, n_splits=5):
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")
    
    # Tambahkan kolom ground truth untuk anomali
    df['anomali_ground_truth'] = False
    df.loc[((df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit)), 'anomali_ground_truth'] = True
    
    X = df.drop(columns=['anomali_ground_truth'])
    y = df['anomali_ground_truth']
    
    # Imputasi nilai hilang
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    scatter_data = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train)

        y_pred_test = model.predict(X_test) == -1

        # Evaluasi pada setiap fold
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_test) * 100)
        metrics['precision'].append(precision_score(y_test, y_pred_test) * 100)
        metrics['recall'].append(recall_score(y_test, y_pred_test) * 100)
        metrics['f1'].append(f1_score(y_test, y_pred_test) * 100)

        # Simpan data scatter plot
        fold_data = X_test.copy()
        fold_data['anomali'] = y_pred_test
        fold_data['fold'] = fold
        scatter_data.append(fold_data)

    # Gabungkan semua data scatter plot dari setiap fold
    scatter_data = pd.concat(scatter_data)

    # Rata-rata metrik evaluasi
    avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}

    return avg_metrics, scatter_data

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
            
            # Deteksi anomali
            try:
                metrics, scatter_data = detect_anomalies(df, TDS_upper_limit, TDS_lower_limit, n_splits=5)
            except ValueError as e:
                st.error(str(e))
                return

            # Menampilkan hasil evaluasi cross-validation
            st.write("Hasil Evaluasi Cross-Validation:")
            st.write(f"Akurasi (rata-rata): {metrics['accuracy']:.2f}%")
            st.write(f"Precision (rata-rata): {metrics['precision']:.2f}%")
            st.write(f"Recall (rata-rata): {metrics['recall']:.2f}%")
            st.write(f"F1 Score (rata-rata): {metrics['f1']:.2f}%")

            # Membuat histogram hasil evaluasi
            st.write("Histogram Metrik Evaluasi:")
            metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
            colors = ['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB']

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metrics_labels, metrics_values, color=colors)
            ax.set_title("Histogram Metrik Evaluasi", fontsize=16)
            ax.set_ylabel("Persentase (%)", fontsize=12)
            ax.set_ylim(0, 100)
            for i, v in enumerate(metrics_values):
                ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10)
            
            st.pyplot(fig)

            # Scatter plot untuk setiap fold
            st.write("Scatter Plot Deteksi Anomali (Setiap Fold):")
            for fold in scatter_data['fold'].unique():
                fold_data = scatter_data[scatter_data['fold'] == fold]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(fold_data[fold_data['anomali'] == False]['pH'], 
                           fold_data[fold_data['anomali'] == False]['TDS'], 
                           color='blue', label='Normal')
                ax.scatter(fold_data[fold_data['anomali'] == True]['pH'], 
                           fold_data[fold_data['anomali'] == True]['TDS'], 
                           color='red', label='Anomali')
                ax.set_xlabel('pH')
                ax.set_ylabel('TDS')
                ax.set_title(f'Scatter Plot Fold {fold + 1}')
                ax.legend()
                st.pyplot(fig)

# Menjalankan halaman utama
main_page()
