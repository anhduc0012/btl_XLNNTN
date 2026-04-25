import json
import os
import pandas as pd
import numpy as np
import torch
import requests
import joblib
from sentence_transformers import SentenceTransformer, util

class Translator:
    def __init__(self):
        self.translator_ready = False
        try:
            import argostranslate.translate
            self.translator_ready = True
        except: pass

    def translate_en_vi(self, text): return self._translate(text, 'en', 'vi')
    def translate_vi_en(self, text): return self._translate(text, 'vi', 'en')

    def _translate(self, text, from_code, to_code):
        if not self.translator_ready or not text: return text
        try:
            import argostranslate.translate
            installed_langs = argostranslate.translate.get_installed_languages()
            from_lang = next((l for l in installed_langs if l.code == from_code), None)
            to_lang   = next((l for l in installed_langs if l.code == to_code), None)
            translation = from_lang.get_translation(to_lang)
            return translation.translate(text)
        except: return text

class NLPEngine:
    def __init__(self, data_dir="d:/New folder/data"):
        self.data_dir = data_dir
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.translator = Translator()
        self.model = SentenceTransformer(self.model_name)
        
        # Nạp mô hình ML chuyên biệt (nếu đã huấn luyện)
        self.classifier = None
        clf_path = os.path.join(self.data_dir, "disease_classifier.pkl")
        if os.path.exists(clf_path):
            try:
                self.classifier = joblib.load(clf_path)
                print("✅ Đã nạp mô hình Classifier chuyên biệt.")
            except: pass
            
        self._load_llm()
        self._load_datasets()
        self._prepare_embeddings()

    def _load_llm(self):
        print("⏳ Đang nạp Local LLM (Qwen2.5-0.5B-Instruct) tối ưu cho cấu hình thấp...")
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self.llm = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                model_kwargs={"torch_dtype": torch.float16},
                device=device
            )
            print("✅ Đã nạp thành công Local LLM.")
        except Exception as e:
            import traceback
            print(f"❌ Lỗi nạp LLM (hãy chạy pip install --upgrade transformers accelerate safetensors): {e}")
            traceback.print_exc()
            self.llm = None

    def _load_datasets(self):
        # 1. Nạp FAQ Tiếng Việt
        try:
            with open(os.path.join(self.data_dir, "faq_vi.json"), 'r', encoding='utf-8') as f:
                self.faq_vi_data = json.load(f)
            self.faq_vi_texts = [item.get("text_to_match", "") for item in self.faq_vi_data]
        except:
            self.faq_vi_data = []; self.faq_vi_texts = []

        # 2. Nạp dữ liệu dinh dưỡng
        try:
            with open(os.path.join(self.data_dir, "nutrition.json"), 'r', encoding='utf-8') as f:
                nut_data = json.load(f)
                self.nutrition_data = nut_data.get("FoundationFoods", [])
        except:
            self.nutrition_data = []

        # 3. Nạp MedQA (Ưu tiên bản Tiếng Việt đã dịch)
        try:
            vi_path = os.path.join(self.data_dir, "medqa_vi.json")
            en_path = os.path.join(self.data_dir, "medqa.json")
            path = vi_path if os.path.exists(vi_path) else en_path
            with open(path, 'r', encoding='utf-8') as f:
                self.medqa_data = json.load(f)
            self.is_medqa_vi = "vi" in os.path.basename(path)
        except:
            self.medqa_data = []; self.is_medqa_vi = False

        # 4. Nạp danh sách triệu chứng (Ưu tiên bản Tiếng Việt)
        try:
            map_path = os.path.join(self.data_dir, "symptom_map_vi.json")
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    self.symptom_map = json.load(f)
                self.symptom_list = list(self.symptom_map.values())
            else:
                df_symp = pd.read_csv(os.path.join(self.data_dir, "symptom.csv"), nrows=0)
                self.symptom_list = [c.replace('_', ' ') for c in df_symp.columns[1:]]
                self.symptom_map = {}
        except:
            self.symptom_list = []; self.symptom_map = {}

    def _prepare_embeddings(self):
        # Tạo embedding cho FAQ VI và MedQA VI (nếu có) để tìm kiếm cực nhanh
        cache_path = os.path.join(self.data_dir, "embeddings_cache.pt")
        if os.path.exists(cache_path):
            try:
                cache_data = torch.load(cache_path)
                if cache_data.get('model_name') == self.model_name:
                    self.faq_vi_embeddings = cache_data.get('faq_vi')
                    self.medqa_embeddings = cache_data.get('medqa_vi')
                    return
            except: pass
        
        print("⏳ Đang tạo vector embedding cho dữ liệu...")
        self.faq_vi_embeddings = self.model.encode(self.faq_vi_texts, convert_to_tensor=True)
        
        # Nếu có MedQA Tiếng Việt, encode mẫu để tìm kiếm semantic
        if self.is_medqa_vi and self.medqa_data:
            medqa_texts = [item.get("question", "") for item in self.medqa_data]
            self.medqa_embeddings = self.model.encode(medqa_texts, convert_to_tensor=True)
        else:
            self.medqa_embeddings = None
            
        torch.save({
            'model_name': self.model_name, 
            'faq_vi': self.faq_vi_embeddings,
            'medqa_vi': self.medqa_embeddings
        }, cache_path)

    def handle_medical_query(self, query_vi: str, chat_history: str = "") -> dict:
        """Sử dụng RAG: Lấy Context từ FAQ/MedQA -> Dùng LLM sinh câu trả lời."""
        
        # Bước 1: Trích xuất triệu chứng và dự đoán bệnh
        found_symptoms = [s for s in self.symptom_list if s.lower() in query_vi.lower()]
        
        predicted_disease = None
        max_prob = 0
        if self.classifier:
            try:
                probs = self.classifier.predict_proba([query_vi])[0]
                idx = np.argmax(probs)
                max_prob = probs[idx]
                predicted_disease = self.classifier.classes_[idx]
            except: pass

        # Bước 2: Tìm kiếm ngữ cảnh (Context)
        context_texts = []
        
        if predicted_disease and max_prob > 0.4:
            context_texts.append(f"Dự đoán ban đầu dựa trên triệu chứng: {predicted_disease}")
            medqa_info = self._search_medqa_optimized(query_vi, predicted_disease)
            if medqa_info:
                context_texts.append(f"Thông tin chuyên sâu (MedQA): {medqa_info}")
                
        # Semantic Search trên FAQ
        query_embedding = self.model.encode(query_vi, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.faq_vi_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        faq_score = cos_scores[best_idx].item()
        
        if faq_score > 0.35:
            res = self.faq_vi_data[best_idx]
            context_texts.append(f"Kiến thức từ FAQ ({res.get('disease', 'Chung')}): {res['answer']}")
            if not predicted_disease and "disease" in res:
                predicted_disease = res["disease"]

        # Search MedQA VI trực tiếp nếu không khớp bệnh
        if self.medqa_embeddings is not None and not context_texts:
            medqa_scores = util.cos_sim(query_embedding, self.medqa_embeddings)[0]
            m_idx = torch.argmax(medqa_scores).item()
            if medqa_scores[m_idx].item() > 0.4:
                res = self.medqa_data[m_idx]
                context_texts.append(f"Tham khảo y khoa: {res['answer']}")

        if not context_texts and not found_symptoms:
            context_texts.append(f"Hãy hỏi người dùng xem họ có các triệu chứng nào trong số này không: {', '.join(self.symptom_list[:5])}")

        context_str = "\\n".join(context_texts)

        # Bước 3: Dùng LLM sinh câu trả lời
        if getattr(self, 'llm', None):
            sys_prompt = "Bạn là Antigravity Medical Bot, trợ lý y khoa AI bằng tiếng Việt. Nhiệm vụ của bạn là tư vấn sức khỏe dựa trên 'Ngữ cảnh được cung cấp' bên dưới. KHÔNG được tự bịa ra thông tin y khoa ngoài ngữ cảnh. Trả lời thân thiện, lịch sự, chuyên nghiệp. Nếu thông tin không đủ, hãy khuyên họ đi khám bác sĩ. Định dạng câu trả lời rõ ràng."
            user_prompt = f"Lịch sử trò chuyện gần đây:\\n{chat_history}\\n\\nNgữ cảnh được cung cấp (từ hệ thống RAG):\\n{context_str}\\n\\nCâu hỏi của người dùng:\\n{query_vi}\\n\\nTrả lời tư vấn:"
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = self.llm(prompt, max_new_tokens=400, do_sample=True, temperature=0.3, top_p=0.9)
                generated_text = outputs[0]["generated_text"][len(prompt):].strip()
                
                # Cleanup model hallucination artifacts if any
                generated_text = generated_text.split("<|im_end|>")[0].strip()
                
                return {"disease": predicted_disease or "Y khoa", "qa": generated_text}
            except Exception as e:
                print(f"Lỗi khi sinh text LLM: {e}")
        
        # Fallback (Nếu không có LLM hoặc lỗi sinh text)
        fallback_ans = f"Chẩn đoán sơ bộ: **{predicted_disease}**.\\n\\n" if predicted_disease else ""
        if context_texts:
            fallback_ans += "\\n\\n".join(context_texts)
        else:
            fallback_ans = "Tôi cần thêm thông tin để hỗ trợ bạn. Bạn có các triệu chứng nào sau đây không: " + ", ".join(self.symptom_list[:5]) + "?"
        
        return {"disease": predicted_disease or "Y khoa", "qa": fallback_ans}

    def _search_medqa_optimized(self, query_vi, disease):
        """Tìm kiếm trong MedQA đã Việt hóa."""
        if not self.medqa_data: return None
        
        # Keyword search nhanh trên MedQA
        keywords = [disease.lower()] + query_vi.lower().split()
        best_match = None
        max_score = 0
        
        for item in self.medqa_data:
            q = item.get("question", "").lower()
            score = sum(3 if kw in q else 0 for kw in keywords)
            if score > max_score:
                max_score = score
                best_match = item
                
        if best_match and max_score >= 3:
            return best_match["answer"]
        return None

    def handle_nutrition_query(self, query_vi: str) -> dict:
        """Xử lý truy vấn dinh dưỡng từ nutrition.json."""
        query_en = query_vi
        if self.llm:
            prompt = f"<|im_start|>system\nYou are a translator. Translate the Vietnamese food name to English. Output ONLY the English name, nothing else. Example: 'thịt bò' -> 'beef', 'táo' -> 'apple'.<|im_end|>\n<|im_start|>user\n{query_vi}<|im_end|>\n<|im_start|>assistant\n"
            try:
                out = self.llm(prompt, max_new_tokens=15, temperature=0.1, do_sample=False)
                result = out[0]['generated_text'].split("<|im_start|>assistant\n")[-1].strip()
                if result: query_en = result
            except: pass
            
        if query_en == query_vi:
            query_en = self.translator.translate_vi_en(query_vi)
            
        query_en = query_en.lower()
        
        import re
        words = re.findall(r'\w+', query_en)
        stop_words = {"how", "many", "calories", "in", "what", "is", "the", "nutrition", "of", "for", "a", "an", "and", "tell", "me", "about", "give", "info", "information", "can", "you", "provide"}
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        best_food = None
        max_score = 0
        best_len = 9999
        
        if keywords:
            for food in self.nutrition_data:
                desc = food.get("description", "").lower()
                # Thưởng điểm nếu khớp nguyên từ, điểm thấp nếu chỉ khớp một phần chuỗi
                score = sum(3 if re.search(rf'\b{kw}\b', desc) else (1 if kw in desc else 0) for kw in keywords)
                
                if score > 0:
                    # Nếu điểm cao hơn HOẶC điểm bằng nhưng tên ngắn hơn (ưu tiên kết quả chung chung, ít phụ gia)
                    if score > max_score or (score == max_score and len(desc) < best_len):
                        max_score = score
                        best_len = len(desc)
                        best_food = food
        
        if best_food:
            name_en = best_food.get("description")
            nutrients = best_food.get("foodNutrients", [])
            
            name_vi = query_vi
            if self.llm:
                prompt = f"<|im_start|>system\nYou are a translator. Translate this English food name to a natural Vietnamese name. Output ONLY the Vietnamese name, no quotes. Example: 'Apple, raw' -> 'Táo', 'Beef, ground' -> 'Thịt bò xay'.<|im_end|>\n<|im_start|>user\n{name_en}<|im_end|>\n<|im_start|>assistant\n"
                try:
                    out = self.llm(prompt, max_new_tokens=15, temperature=0.1, do_sample=False)
                    result = out[0]['generated_text'].split("<|im_start|>assistant\n")[-1].strip()
                    if result: name_vi = result.capitalize()
                except: pass
            
            res_text = f"Thông tin dinh dưỡng cho **{name_vi}** (trên 100g):\n\n"
            important = {"Energy": "Năng lượng", "Protein": "Đạm", "Total lipid (fat)": "Chất béo", 
                         "Carbohydrate, by difference": "Tinh bột", "Fiber, total dietary": "Chất xơ"}
            
            for nut in nutrients:
                n_name = nut["nutrient"]["name"]
                if n_name in important:
                    res_text += f"- **{important[n_name]}**: {nut['amount']} {nut['nutrient']['unit_name'] if 'unit_name' in nut['nutrient'] else nut['nutrient'].get('unitName', '')}\n"
            
            return {"qa": res_text}
        
        return {"qa": "Tôi chưa tìm thấy thông tin dinh dưỡng cho thực phẩm này. Bạn có thể thử hỏi tên tiếng Anh của nó (ví dụ: 'Apple', 'Beef') để tôi tìm kiếm chính xác hơn."}

    def handle_chit_chat(self, query: str, chat_history: str = "") -> str:
        """Xử lý giao tiếp thông thường bằng LLM (nếu có) hoặc Rule-based."""
        if getattr(self, 'llm', None):
            sys_prompt = "Bạn là Antigravity Medical Bot, một trợ lý y khoa AI bằng tiếng Việt. Hãy trả lời ngắn gọn, vui vẻ, lịch sự với các câu giao tiếp thông thường của người dùng."
            user_prompt = f"Lịch sử trò chuyện:\\n{chat_history}\\n\\nNgười dùng nói: {query}\\n\\nTrả lời:"
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            try:
                prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = self.llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=0.9)
                return outputs[0]["generated_text"][len(prompt):].split("<|im_end|>")[0].strip()
            except:
                pass

        # Fallback rule-based
        query = query.lower()
        responses = {
            "chào": "Xin chào! Tôi là trợ lý y tế thông minh. Tôi có thể giúp gì cho bạn?",
            "tên gì": "Tôi là Antigravity Medical Bot, phiên bản Offline hỗ trợ y tế và dinh dưỡng.",
            "cảm ơn": "Rất vui được hỗ trợ bạn. Chúc bạn nhiều sức khỏe!",
            "tạm biệt": "Chào tạm biệt bạn. Hãy giữ gìn sức khỏe nhé!",
            "thông minh": "Cảm ơn bạn! Tôi luôn cố gắng học hỏi từ dữ liệu y khoa của bạn.",
            "hello": "Hi there! How can I help you with your health today?",
            "bye": "Goodbye! Take care of yourself."
        }
        for key in responses:
            if key in query: return responses[key]
        return "Chào bạn! Tôi đang lắng nghe, bạn cần tư vấn gì về y tế hay dinh dưỡng không?"

    def execute_task_location(self, query: str) -> dict:
        # ... (giữ nguyên logic OSM hiện tại)
        facility_type = "hospital"
        label = "bệnh viện"
        if "nhà thuốc" in query or "thuốc" in query: facility_type = "pharmacy"; label = "nhà thuốc"
        elif "phòng khám" in query: facility_type = "clinic"; label = "phòng khám"

        try:
            url = f"https://nominatim.openstreetmap.org/search?q={facility_type}+vietnam&format=json&limit=5"
            headers = {'User-Agent': 'AntigravityMedicalBot/1.1'}
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            if data:
                result_text = f"Dưới đây là một số **{label}** tại Việt Nam:\n"
                locations = []
                for i, loc in enumerate(data):
                    name = loc.get('display_name', '').split(',')[0]
                    result_text += f"{i+1}. **{name}**\n"
                    locations.append({"name": name, "lat": loc.get('lat'), "lon": loc.get('lon')})
                return {"qa": result_text, "map_data": locations}
        except: pass
        return {"qa": f"Không thể kết nối tới bản đồ. Bạn hãy tìm **{label}** trên Google Maps nhé."}

engine = NLPEngine()
