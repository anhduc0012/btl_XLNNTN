import pandas as pd
import json
import os
import argostranslate.package
import argostranslate.translate

def get_translator():
    from_code = "en"
    to_code = "vi"
    # Đảm bảo đã có gói ngôn ngữ
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
    )
    argostranslate.package.install_from_path(package_to_install.download())
    return argostranslate.translate

def translate_symptoms(data_dir):
    print("⏳ Đang dịch danh sách triệu chứng...")
    df = pd.read_csv(os.path.join(data_dir, "symptom.csv"), nrows=0)
    en_headers = df.columns.tolist()
    
    # Dịch từng header (trừ cột đầu tiên là diseases)
    vi_headers = ["diseases"]
    for h in en_headers[1:]:
        clean_h = h.replace('_', ' ')
        vi_h = argostranslate.translate.translate(clean_h, "en", "vi")
        vi_headers.append(vi_h)
        print(f"  {clean_h} -> {vi_h}")
    
    # Lưu mapping để sử dụng trong engine
    with open(os.path.join(data_dir, "symptom_map_vi.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(en_headers, vi_headers)), f, ensure_ascii=False, indent=2)
    print("✅ Đã lưu bản đồ triệu chứng tiếng Việt!")

def translate_medqa_sample(data_dir, limit=1000):
    print(f"⏳ Đang dịch {limit} câu hỏi MedQA đầu tiên...")
    with open(os.path.join(data_dir, "medqa.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vi_data = []
    for i, item in enumerate(data[:limit]):
        q_vi = argostranslate.translate.translate(item['Question'], "en", "vi")
        a_vi = argostranslate.translate.translate(item['Answer'], "en", "vi")
        vi_data.append({
            "question": q_vi,
            "answer": a_vi,
            "disease": item.get("Disease", "Y khoa")
        })
        if i % 10 == 0: print(f"  Đã dịch {i}/{limit}...")
    
    with open(os.path.join(data_dir, "medqa_vi.json"), "w", encoding="utf-8") as f:
        json.dump(vi_data, f, ensure_ascii=False, indent=2)
    print("✅ Đã tạo medqa_vi.json!")

if __name__ == "__main__":
    DATA_PATH = "d:/New folder/data"
    translate_symptoms(DATA_PATH)
    translate_medqa_sample(DATA_PATH, limit=500) # Dịch thử 500 câu trước

