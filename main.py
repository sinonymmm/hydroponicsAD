import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    return df

# Fungsi untuk menjalankan cross-validation
def cross_validation_with_isolation_forest(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train_imputed)

        y_pred = model.predict(X_test_imputed)
        y_pred = (y_pred == -1)

        metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
        metrics['Precision'].append(precision_score(y_test, y_pred) * 100)
        metrics['Recall'].append(recall_score(y_test, y_pred) * 100)
        metrics['F1 Score'].append(f1_score(y_test, y_pred) * 100)

    return metrics

# Fungsi utama untuk deteksi anomali
def detect_anomalies_with_visualization(df, TDS_upper_limit, TDS_lower_limit):
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")

    df['anomali_ground_truth'] = ((df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit))

    # Cross-validation
    metrics = cross_validation_with_isolation_forest(df[['pH', 'TDS']], df['anomali_ground_truth'])

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[df['anomali_ground_truth'] == False]['pH'], df[df['anomali_ground_truth'] == False]['TDS'], color='blue', label='Normal')
    ax.scatter(df[df['anomali_ground_truth'] == True]['pH'], df[df['anomali_ground_truth'] == True]['TDS'], color='red', label='Anomali')
    ax.set_xlabel('pH')
    ax.set_ylabel('TDS')
    ax.set_title('Scatter Plot Data dengan Anomali')
    ax.legend()

    return fig, metrics

# Fungsi utama untuk Streamlit
def main_page():
    st.write("Selamat datang di Aplikasi Deteksi Anomali Nutrisi Air Hidroponik dengan Cross-Validation")

    uploaded_file = st.file_uploader("Unggah file dataset CSV", type="csv")

    TDS_upper_limit = st.number_input("Batas atas TDS:", value=1200)
    TDS_lower_limit = st.number_input("Batas bawah TDS:", value=400)

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)

        try:
            fig, metrics = detect_anomalies_with_visualization(df, TDS_upper_limit, TDS_lower_limit)

            st.write("Visualisasi Scatter Plot:")
            st.pyplot(fig)

            st.write("Histogram Metrik Evaluasi Cross-Validation:")
            metric_names = list(metrics.keys())
            metric_values = [sum(metrics[m]) / len(metrics[m]) for m in metric_names]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metric_names, metric_values, color=['#FF6F61', '#6BAED6', '#FFD700', '#8DA0CB'])
            ax.set_title('Histogram Metrik Evaluasi')
            ax.set_ylabel('Persentase (%)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)

        except ValueError as e:
            st.error(str(e))

main_page()
