"""
Script xây dựng tri thức độc quyền: Chỉ sử dụng ViMedical_Disease.csv, symptom.csv và medqa.json.
"""

import os
import json
import pandas as pd
import sys

# Thêm đường dẫn để import Translator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
try:
    from nlp_engine import Translator
    translator = Translator()
except:
    translator = None

CSV_PATH        = r"d:\New folder\data\ViMedical_Disease.csv"
SYMPTOM_PATH    = r"d:\New folder\data\symptom.csv"
MEDQA_PATH      = r"d:\New folder\data\medqa.json"
OUTPUT_PATH     = r"d:\New folder\data\faq_vi.json"

def build_faq():
    faq = []
    
    # 1. Phân tích symptom.csv để lấy danh sách triệu chứng cho mỗi bệnh
    print("--- Đang nạp triệu chứng từ symptom.csv ---")
    disease_to_symptoms = {}
    try:
        df_symp = pd.read_csv(SYMPTOM_PATH, nrows=10000) # Load mẫu lớn hơn
        cols = df_symp.columns[1:]
        for _, row in df_symp.iterrows():
            dname = str(row['diseases']).strip().lower()
            active_symps = [cols[i] for i, val in enumerate(row[1:]) if val == 1]
            if active_symps:
                disease_to_symptoms[dname] = active_symps
    except Exception as e:
        print(f"Lỗi symptom.csv: {e}")

    # 2. Phân tích medqa.json để lấy kiến thức bổ trợ
    print("--- Đang nạp kiến thức từ medqa.json ---")
    medqa_knowledge = {} # disease_name -> sample_answer
    try:
        with open(MEDQA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[:2000]: # Lấy 2000 mẫu đầu
                q = item.get("Question", "").lower()
                # Thử tìm tên bệnh trong câu hỏi của MedQA để map
                for d in disease_to_symptoms.keys():
                    if d in q:
                        medqa_knowledge[d] = item.get("Answer", "")
                        break
    except: pass

    # 3. Hợp nhất vào ViMedical_Disease.csv
    print("--- Đang xây dựng bộ FAQ từ ViMedical_Disease.csv ---")
    try:
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            d_name = str(row['Disease']).strip()
            q_text = str(row['Question']).strip()
            
            d_key = d_name.lower()
            symps = disease_to_symptoms.get(d_key, [])
            med_info = medqa_knowledge.get(d_key, "")
            
            # Việt hóa triệu chứng nếu có translator
            if translator and symps:
                symps = [translator.translate_en_vi(s) for s in symps[:10]]
            
            # Tạo câu trả lời tổng hợp
            ans = f"### Bệnh lý xác định: **{d_name}**\n\n"
            if symps:
                ans += "**Các triệu chứng điển hình:**\n- " + "\n- ".join(symps) + "\n\n"
            
            if med_info:
                if translator: med_info = translator.translate_en_vi(med_info)
                ans += f"**Thông tin chuyên khoa (MedQA):**\n{med_info}\n\n"
            
            ans += "⚠️ *Thông tin chỉ mang tính chất tham khảo dựa trên dữ liệu hiện có.*"

            faq.append({
                "text_to_match": q_text,
                "disease": d_name,
                "answer": ans
            })
    except Exception as e:
        print(f"Lỗi build: {e}")

    return faq

def main():
    faq = build_faq()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(faq, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã cập nhật tri thức từ 3 nguồn độc lập. Tổng số: {len(faq)} mục.")

if __name__ == "__main__":
    main()
