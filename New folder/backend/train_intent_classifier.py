import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Dữ liệu huấn luyện mẫu (Bilingual: Vietnamese & English)
data = [
    # Task Location
    ("bệnh viện gần nhất", "task_location"),
    ("nhà thuốc ở đâu", "task_location"),
    ("phòng khám gần đây", "task_location"),
    ("địa chỉ trạm y tế", "task_location"),
    ("tìm bệnh viện quanh đây", "task_location"),
    ("mách tôi chỗ mua thuốc", "task_location"),
    ("hospital near me", "task_location"),
    ("pharmacy address", "task_location"),
    ("where is the nearest clinic", "task_location"),
    ("find medical center", "task_location"),
    
    # Nutrition Query
    ("calo trong quả táo", "nutrition_query"),
    ("dinh dưỡng trong thịt bò", "nutrition_query"),
    ("ăn gì tốt cho tim mạch", "nutrition_query"),
    ("vitamin c có trong thực phẩm nào", "nutrition_query"),
    ("chế độ ăn giảm cân", "nutrition_query"),
    ("thông tin dinh dưỡng chuối", "nutrition_query"),
    ("calories in banana", "nutrition_query"),
    ("nutrition facts apple", "nutrition_query"),
    ("what to eat for muscles", "nutrition_query"),
    ("diet for diabetes", "nutrition_query"),
    
    # Chit-chat
    ("chào bạn", "chit_chat"),
    ("hello", "chit_chat"),
    ("hi", "chit_chat"),
    ("bạn là ai", "chit_chat"),
    ("tên bạn là gì", "chit_chat"),
    ("cảm ơn", "chit_chat"),
    ("tạm biệt", "chit_chat"),
    ("who are you", "chit_chat"),
    ("what is your name", "chit_chat"),
    ("thanks", "chit_chat"),
    ("bye", "chit_chat"),
    ("bạn thông minh quá", "chit_chat"),
    
    # Symptom Query
    ("tôi bị đau đầu", "symptom_query"),
    ("ho và sốt nhẹ", "symptom_query"),
    ("nhức mỏi chân tay", "symptom_query"),
    ("biểu hiện của bệnh cúm", "symptom_query"),
    ("triệu chứng sốt xuất huyết", "symptom_query"),
    ("đau bụng quặn thắt", "symptom_query"),
    ("i have a headache", "symptom_query"),
    ("fever and cough", "symptom_query"),
    ("symptoms of cold", "symptom_query"),
    ("stomach ache", "symptom_query"),
    
    # Medical Query (QA)
    ("cách phòng tránh covid", "medical_query"),
    ("điều trị tiểu đường", "medical_query"),
    ("bệnh cao huyết áp là gì", "medical_query"),
    ("vắc xin sởi có tác dụng gì", "medical_query"),
    ("làm sao để ngủ ngon", "medical_query"),
    ("how to prevent flu", "medical_query"),
    ("treatment for cancer", "medical_query"),
    ("what is diabetes", "medical_query"),
    ("heart disease prevention", "medical_query")
]

X = [item[0] for item in data]
y = [item[1] for item in data]

def train_and_save():
    model_path = r"d:\New folder\data\intent_classifier_model.pkl"
    
    # Xây dựng Pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print("--- Training Intent Classifier ---")
    pipeline.fit(X, y)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    train_and_save()
