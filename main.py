import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    return df

# Fungsi untuk deteksi anomali dengan visualisasi
def detect_anomalies_with_visualization(df, TDS_upper_limit, TDS_lower_limit):
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")

    # Menandai anomali berdasarkan ground truth
    df['anomali_ground_truth'] = ((df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit))

    # Split data menjadi train dan test
    X = df[['pH', 'TDS']]
    y = df['anomali_ground_truth']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)

    # Prediksi
    y_pred_train = model.predict(X_train) == -1
    y_pred_test = model.predict(X_test) == -1

    # Evaluasi metrik
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_test) * 100,
        'Precision': precision_score(y_test, y_pred_test) * 100,
        'Recall': recall_score(y_test, y_pred_test) * 100,
        'F1 Score': f1_score(y_test, y_pred_test) * 100,
    }

    # Scatter plot untuk data train
    st.write("Scatter Plot Data Train:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train[y_train == False]['pH'], X_train[y_train == False]['TDS'], color='blue', label='Normal')
    ax.scatter(X_train[y_pred_train]['pH'], X_train[y_pred_train]['TDS'], color='red', label='Anomali')
    ax.set_xlabel('pH')
    ax.set_ylabel('TDS')
    ax.set_title('Scatter Plot Data Train')
    ax.legend()
    st.pyplot(fig)

    # Scatter plot untuk data test
    st.write("Scatter Plot Data Test:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test[y_test == False]['pH'], X_test[y_test == False]['TDS'], color='blue', label='Normal')
    ax.scatter(X_test[y_pred_test]['pH'], X_test[y_pred_test]['TDS'], color='red', label='Anomali')
    ax.set_xlabel('pH')
    ax.set_ylabel('TDS')
    ax.set_title('Scatter Plot Data Test')
    ax.legend()
    st.pyplot(fig)

    return metrics

# Fungsi utama untuk Streamlit
def main_page():
    st.title("Deteksi Anomali Nutrisi Air Hidroponik")
    st.write("Unggah dataset untuk mendeteksi anomali pada data nutrisi air.")

    uploaded_file = st.file_uploader("Unggah file dataset CSV", type="csv")

    TDS_upper_limit = st.number_input("Batas atas TDS:", value=1200)
    TDS_lower_limit = st.number_input("Batas bawah TDS:", value=400)

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)

        try:
            metrics = detect_anomalies_with_visualization(df, TDS_upper_limit, TDS_lower_limit)

            # Histogram metrik evaluasi
            st.write("Histogram Metrik Evaluasi:")
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metric_names, metric_values, color=['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB'])
            ax.set_title('Histogram Metrik Evaluasi')
            ax.set_ylabel('Persentase (%)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

        except ValueError as e:
            st.error(str(e))

main_page()
