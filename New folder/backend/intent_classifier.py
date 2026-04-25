import re
import joblib
import os

# Load model global to avoid reloading on every call
MODEL_PATH = r"d:\New folder\data\intent_classifier_model.pkl"
intent_model = None
if os.path.exists(MODEL_PATH):
    try:
        intent_model = joblib.load(MODEL_PATH)
    except: pass

def get_intent(text):
    text = text.lower().strip()
    
    # Ưu tiên sử dụng Machine Learning Classifier
    if intent_model:
        try:
            probs = intent_model.predict_proba([text])[0]
            max_prob = max(probs)
            intent = intent_model.classes_[probs.tolist().index(max_prob)]
            
            # Nếu độ tin cậy cao (>0.5), trả về intent từ ML
            if max_prob > 0.5:
                return intent
        except: pass

    # Fallback: Quy tắc từ khóa (Rule-based) nếu ML không tự tin hoặc chưa nạp được
    
    # 1. Nhận diện tác vụ: Tìm kiếm vị trí (Task Location)
    location_keywords = ['bệnh viện', 'nhà thuốc', 'phòng khám', 'trạm y tế', 'ở đâu', 'gần đây', 'địa chỉ', 'hospital', 'pharmacy', 'clinic', 'where']
    if any(kw in text for kw in location_keywords) and ('gần' in text or 'đâu' in text or 'near' in text or 'where' in text):
        return "task_location"
    
    # 2. Nhận diện truy vấn dinh dưỡng (Nutrition Query)
    nutrition_keywords = ['dinh dưỡng', 'calo', 'vitamin', 'thực phẩm', 'ăn gì', 'ăn uống', 'nutrient', 'calories', 'food', 'eat', 'diet', 'bổ sung']
    if any(kw in text for kw in nutrition_keywords):
        return "nutrition_query"

    # 3. Nhận diện giao tiếp (Chit-chat)
    chitchat_keywords = ['chào', 'hello', 'hi', 'tạm biệt', 'cảm ơn', 'thông minh', 'là ai', 'tên gì', 'bye', 'thanks', 'who are you', 'what is your name']
    if any(text.startswith(kw) for kw in chitchat_keywords) or len(text.split()) < 2:
        return "chit_chat"
    
    # 4. Nhận diện triệu chứng (Symptom Query)
    symptom_indicators = ['bị', 'đau', 'nhức', 'mỏi', 'ho', 'sốt', 'triệu chứng', 'biểu hiện', 'symptom', 'pain', 'ache', 'fever', 'cough']
    if any(kw in text for kw in symptom_indicators):
        return "symptom_query"

    # 5. Mặc định là truy vấn y tế (Medical QA)
    return "medical_query"
