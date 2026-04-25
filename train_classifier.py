import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CSV_PATH = r"d:\New folder\data\ViMedical_Disease.csv"
MODEL_PATH = r"d:\New folder\data\disease_classifier.pkl"

def train():
    print("--- HUẤN LUYỆN LẠI MÔ HÌNH VỚI ĐỘ CHÍNH XÁC CAO (LOGISTIC REGRESSION) ---")
    
    if not os.path.exists(CSV_PATH):
        print("❌ Không tìm thấy file dữ liệu.")
        return
        
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['Question', 'Disease'])
    
    X = df['Question']
    y = df['Disease']
    
    # Sử dụng LogisticRegression để có thể lấy xác suất (predict_proba)
    # Tăng max_iter để đảm bảo hội tụ với dữ liệu lớn
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, stop_words=None)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial'))
    ])
    
    print(f"Đang học {len(X)} mẫu câu hỏi...")
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Đã huấn luyện xong mô hình có độ tin cậy! Lưu tại: {MODEL_PATH}")

if __name__ == "__main__":
    train()
