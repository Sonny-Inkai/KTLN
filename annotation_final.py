import pandas as pd
import google.generativeai as genai
import os
import json
import time
import traceback
# --- 1. CẤU HÌNH---
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDFkjWd2y_v7nk_liZtnTvO8oZ-1wFH52Q")
#("GOOGLE_API_KEY", "AIzaSyDFkjWd2y_v7nk_liZtnTvO8oZ-1wFH52Q")
#("GOOGLE_API_KEY", "AIzaSyCm_sPu4X7pZ2BHLd5_l17km2JI927pH40")
INPUT_FILE = "C:\\Users\\Tuong\\Downloads\\cleaned_Data_new.xlsx"
OUTPUT_FILE = 'results_simple_format.json'
PROCESSED_IDS_FILE = 'history_simple_format.txt'
ID_COLUMN = 'id'
TITLE_COLUMN = 'title'
CONTEXT_COLUMN = 'context'
# Setting cho mỗi lần chạy
MAX_ITEMS_PER_RUN = 100
SLEEP_TIME_SECONDS = 5

# --- 2. KIỂM TRA SETTING VÀ TẠO API ---
if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    exit()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"LỖI: Không thể cấu hình Gemini API: {e}")
    exit()

# --- 3. ĐỊNH NGHĨA PROMPT ---
ENTITY_DEFINITIONS = """
- ORGANIZATION: Tên công ty, tổ chức, cơ quan nhà nước
- LOCATION: Địa danh địa lý
- DATE/TIME: Ngày, giờ, khoảng thời gian cụ thể
- LEGAL_PROVISION: Điều khoản, quy định, luật, nghị định
"""
RELATION_DEFINITIONS = """
- Effective_From (Có hiệu lực từ): LEGAL_PROVISION -> DATE/TIME
- Applicable_In (Áp dụng tại): LEGAL_PROVISION -> LOCATION
- Relates_To (Liên quan đến): [LỰA CHỌN CUỐI CÙNG] LEGAL_PROVISION -> ORGANIZATION/LEGAL_PROVISION
- Amended_By (Sửa đổi bởi): LEGAL_PROVISION -> LEGAL_PROVISION
"""
SELF_VERIFY_PROMPT_TEMPLATE = """
You are a meticulous legal data analyst robot. Your task is to analyze the Vietnamese text, extract relationships following strict priority rules, and verify your own work.

--- 1. ENTITY DEFINITIONS ---
{entity_definitions}

--- 2. RELATION DEFINITIONS (Strict) ---
{relation_definitions}

--- 3. [CRITICAL] PRIORITY RULES ---
1.  **ƯU TIÊN CÁC NHÃN CỤ THỂ**: Luôn luôn kiểm tra xem `Effective_From`, `Applicable_In`, hoặc `Amended_By` có áp dụng được không TRƯỚC KHI xem xét `Relates_To`.
2.  **`Relates_To` LÀ LỰA CHỌN CUỐI CÙNG**: Chỉ dùng `Relates_To` nếu không có nhãn nào khác phù hợp.

--- 4. OUTPUT FORMAT AND RULES ---
- Output MUST be plain text. Each relation on a new line.
- Format: <HEAD_LABEL> head_text <TAIL_LABEL> tail_text <RELATION_LABEL>
- Extract text EXACTLY as it appears in the source.
- **[CRITICAL] NO EXTRA TEXT**: Your entire response MUST NOT contain any explanations, greetings, introductions, or summaries. If no relations are found, provide a completely empty response.

--- 5. EXAMPLES (Good and Bad) ---
-- GOOD EXAMPLE --
CONTEXT: Điều 25: Hiệu lực thi hành. Quy chế này tham chiếu theo Luật Doanh nghiệp 2020 và có hiệu lực từ ngày 01/12/2025 tại Việt Nam.
EXPECTED OUTPUT:
<LEGAL_PROVISION> Quy chế này <LEGAL_PROVISION> Luật Doanh nghiệp 2020 <Relates_To>
<LEGAL_PROVISION> Quy chế này <DATE/TIME> ngày 01/12/2025 <Effective_From>
<LEGAL_PROVISION> Quy chế này <LOCATION> Việt Nam <Applicable_In>

-- BAD EXAMPLE (Incorrect Logic) --
CONTEXT: Quyết định này có hiệu lực kể từ ngày ký.
WRONG OUTPUT: <LEGAL_PROVISION> Quyết định này <DATE/TIME> ngày ký <Relates_To>
CORRECT OUTPUT: <LEGAL_PROVISION> Quyết định này <DATE/TIME> ngày ký <Effective_From>

--- 6. SELF-VERIFICATION PROCESS ---
Before providing the final output, perform these mental checks:
1.  **Analyze**: Read the text and identify potential relations.
2.  **Prioritize & Verify**: Did I follow the PRIORITY RULES? Have I wrongly used `Relates_To` where a more specific label fits?
3.  **Format**: Present ONLY the verified relations in the required plain text format.

--- 7. TEXT TO ANALYZE (ID: {item_id}) ---
{text_context}
--- END OF TEXT ---

**FINAL INSTRUCTION: Your output must only be the data lines. Do not say "Here are the results" or anything similar. Output only the data.**
Plain Text Output:
"""

# --- 4. GỌI API ---
def get_relations_from_text(text_context, item_id):
    prompt = SELF_VERIFY_PROMPT_TEMPLATE.format(
        entity_definitions=ENTITY_DEFINITIONS,
        relation_definitions=RELATION_DEFINITIONS,
        item_id=item_id,
        text_context=text_context
    )
    
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generation_config = genai.types.GenerationConfig(temperature=0.0)
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        if response.prompt_feedback:
            print(f"      [Lỗi] Lệnh gọi cho {item_id} bị chặn. Lý do: {response.prompt_feedback}")
            return None
        return ""
    except Exception as e:
        print(f"      [Lỗi] Khi gọi Gemini API cho {item_id}: {e}")
        return None

# --- 5. LOGIC  ---
def main():
    all_results = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f: all_results = json.load(f)
            print(f"Đã đọc {len(all_results)} kết quả từ '{OUTPUT_FILE}'.")
        except Exception as e: print(f"Lỗi đọc file JSON '{OUTPUT_FILE}': {e}.")
    
    processed_items_set = set()
    if os.path.exists(PROCESSED_IDS_FILE):
        try:
            with open(PROCESSED_IDS_FILE, 'r', encoding='utf-8') as f: processed_items_set = {line.strip() for line in f if line.strip()}
            print(f"Đã đọc {len(processed_items_set)} ID đã xử lý từ '{PROCESSED_IDS_FILE}'.")
        except Exception as e: print(f"Lỗi đọc file lịch sử '{PROCESSED_IDS_FILE}': {e}")

    try:
        print(f"Đang đọc dữ liệu từ '{INPUT_FILE}'...")
        df = pd.read_excel(INPUT_FILE, header=0)
        # Kiểm tra các cột
        required_columns = [ID_COLUMN, TITLE_COLUMN, CONTEXT_COLUMN]
        if not all(col in df.columns for col in required_columns):
            print(f"LỖI: File Excel thiếu các cột cần thiết: {required_columns}.")
            return
        # Bỏ qua các hàng có giá trị rỗng
        df.dropna(subset=required_columns, inplace=True)
        df['item_id'] = df[ID_COLUMN].astype(str)
        
        data_unprocessed = df[~df['item_id'].isin(processed_items_set)]
        total_unprocessed = len(data_unprocessed)
        print(f"Tổng số mục hợp lệ trong file: {len(df)}")
        print(f"Tổng số mục chưa xử lý: {total_unprocessed}")

        if total_unprocessed == 0:
            print("Tất cả các mục đã được xử lý.")
            return
        data_to_process_this_run = data_unprocessed.head(MAX_ITEMS_PER_RUN)
        actual_items_this_run = len(data_to_process_this_run)
        print(f"\nBắt đầu xử lý {actual_items_this_run} mục trong lần chạy này...")
        processed_count, error_count, newly_processed_ids = 0, 0, []
        for index, row in data_to_process_this_run.iterrows():
            item_id = row['item_id']
            title_to_use = str(row[TITLE_COLUMN]).strip()
            context_to_send = str(row[CONTEXT_COLUMN]).strip()
            processed_count += 1
            newly_processed_ids.append(item_id)
            print(f"\n[{processed_count}/{actual_items_this_run}] Đang xử lý: {item_id} ({title_to_use})")
            # Gọi API cho từng mục\
            result_text = get_relations_from_text(context_to_send, item_id)
            if result_text is None:
                error_count += 1
            elif result_text:
                num_relations = len(result_text.split('\n'))
                print(f"   => Thành công: Tìm thấy {num_relations} quan hệ.")
                # Lưu kết quả
                all_results[item_id] = {
                    "title": title_to_use,
                    "input_text": context_to_send,
                    "extracted_relations": result_text
                }
            else:
                print("   => Không tìm thấy quan hệ nào.")
            time.sleep(SLEEP_TIME_SECONDS)
        print(f"Tổng cộng đã xử lý: {processed_count}, trong đó có {error_count} lỗi.")
        if processed_count > 0:
            try:
                print(f"Đang lưu {len(all_results)} kết quả vào '{OUTPUT_FILE}'...")
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                print("Lưu file JSON thành công.")
            except Exception as e:
                print(f"LỖI KHI LƯU FILE JSON: {e}")
            try:
                print(f"Đang cập nhật {len(newly_processed_ids)} ID vào file lịch sử '{PROCESSED_IDS_FILE}'...")
                with open(PROCESSED_IDS_FILE, 'a', encoding='utf-8') as f:
                    for item_id in newly_processed_ids:
                        f.write(item_id + '\n')
                print("Cập nhật file lịch sử thành công.")
            except Exception as e:
                print(f"LỖI KHI CẬP NHẬT FILE LỊCH SỬ: {e}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file đầu vào '{INPUT_FILE}'. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn: {e}")
        traceback.print_exc()
    print("\nScript đã thực thi xong.")
if __name__ == "__main__":
    main()