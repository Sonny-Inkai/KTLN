import sentencepiece as spm
import os
from transformers import T5Tokenizer
import shutil

# --- 1. THIẾT LẬP CÁC THAM SỐ ---
print("--- BƯỚC 1: THIẾT LẬP THAM SỐ ---")

# Đường dẫn đến file dữ liệu văn bản thô
DATA_DIR = "." 
TRAIN_FILE = os.path.join(DATA_DIR, "dataset.txt")

# Tên file tạm thời cho mô hình sentencepiece
MODEL_PREFIX = "temp_vietnamese_legal_tokenizer"

# Thư mục cuối cùng chứa tokenizer hoàn chỉnh theo chuẩn Hugging Face
FINAL_HF_TOKENIZER_DIR = "vietnamese_legal_hf_tokenizer"

# --- 2. ĐỊNH NGHĨA CÁC TOKEN ĐẶC BIỆT ---
print("--- BƯỚC 2: ĐỊNH NGHĨA TOKEN ĐẶC BIỆT ---")

# Các token đặc biệt cho lĩnh vực pháp luật
domain_special_tokens = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
    "<Relates_To>", "<Amended_By>"
]
# Các token <extra_id_..> cần thiết cho T5
extra_id_tokens = [f"<extra_id_{i}>" for i in range(100)]

# Ghép tất cả vào một danh sách duy nhất
user_defined_symbols = domain_special_tokens 
print(f"Tổng số token đặc biệt cần thêm: {len(user_defined_symbols)}")

# --- 3. CẤU HÌNH VÀ HUẤN LUYỆN SENTENCEPIECE ---
print("\n--- BƯỚC 3: HUẤN LUYỆN SENTENCEPIECE ---")

# Kích thước từ vựng MONG MUỐN cho các token thông thường
BASE_VOCAB_SIZE = 10000 
# Kích thước từ vựng CUỐI CÙNG = kích thước cơ bản + số lượng token đặc biệt
# FINAL_VOCAB_SIZE = BASE_VOCAB_SIZE + len(user_defined_symbols)
MODEL_TYPE = "bpe"

train_args = [
    f"--input={TRAIN_FILE}",
    f"--model_prefix={MODEL_PREFIX}",
    f"--vocab_size={BASE_VOCAB_SIZE}",
    f"--model_type={MODEL_TYPE}",
    f"--user_defined_symbols={','.join(user_defined_symbols)}", # "Bake" token vào thẳng model

    # Higher character coverage for Vietnamese
    "--character_coverage=1",
    "--pad_id=0",
    "--unk_id=1", 
    "--bos_id=2",
    "--eos_id=3",
    "--pad_piece=<pad>",
    "--unk_piece=<unk>",
    "--bos_piece=<s>",
    "--eos_piece=</s>",
    # Better normalization for Vietnamese
    "--normalization_rule_name=nmt_nfkc",
    "--add_dummy_prefix=true",
    "--remove_extra_whitespaces=true", # chưa biết nữa
    "--split_by_whitespace=true",             
    "--split_by_number=false",                
    "--shuffle_input_sentence=true",          
    "--byte_fallback=true", # tr
]

print(f"Kích thước từ vựng mục tiêu: {BASE_VOCAB_SIZE}")
try:
    spm.SentencePieceTrainer.Train(" ".join(train_args))
    print(f"Huấn luyện SentencePiece thành công! Đã tạo file: {MODEL_PREFIX}.model")
except Exception as e:
    print(f"LỖI: Huấn luyện SentencePiece thất bại: {e}")
    exit(1)

# --- 4. CHUYỂN ĐỔI SANG ĐỊNH DẠNG HUGGING FACE (PHƯƠNG PHÁP AN TOÀN) ---
print("\n--- BƯỚC 4: TẠO TOKENIZER CHUẨN HUGGING FACE ---")
# Đây là phương pháp an toàn và được khuyên dùng nhất.
# Nó sẽ tự động tạo ra các file config chuẩn, tránh mọi lỗi lầm thủ công.

try:
    # Tải tokenizer từ file .model vừa tạo.
    # `keep_spm_overrides=True` là CỰC KỲ QUAN TRỌNG.
    # Thêm `legacy=False` để ngăn T5Tokenizer tự động thêm các extra_id tokens một lần nữa.
    spm_model_path = f"{MODEL_PREFIX}.model"
    tokenizer = T5Tokenizer(vocab_file=spm_model_path, keep_spm_overrides=True, legacy=False)

    # Xóa thư mục cũ nếu tồn tại và tạo thư mục mới
    if os.path.exists(FINAL_HF_TOKENIZER_DIR):
        shutil.rmtree(FINAL_HF_TOKENIZER_DIR)
    os.makedirs(FINAL_HF_TOKENIZER_DIR)

    # Lưu lại tokenizer vào thư mục mới. Thư viện sẽ tự tạo tất cả các file cần thiết.
    tokenizer.save_pretrained(FINAL_HF_TOKENIZER_DIR)
    print(f"Đã lưu tokenizer hoàn chỉnh vào thư mục: '{FINAL_HF_TOKENIZER_DIR}'")
except Exception as e:
    print(f"LỖI: Không thể tạo tokenizer Hugging Face: {e}")
    exit(1)

# --- 5. KIỂM TRA KẾT QUẢ CUỐI CÙNG ---
print("\n--- BƯỚC 5: KIỂM TRA KẾT QUẢ ---")
try:
    final_tokenizer = T5Tokenizer.from_pretrained(FINAL_HF_TOKENIZER_DIR)
    actual_vocab_size = len(final_tokenizer)
    
    print(f"Kích thước từ vựng mong muốn: {BASE_VOCAB_SIZE}")
    print(f"Kích thước từ vựng thực tế:  {actual_vocab_size}")

    if actual_vocab_size == BASE_VOCAB_SIZE:
        print("-> TUYỆT VỜI! Kích thước từ vựng đã chính xác.")
    else:
        print("-> LỖI! Kích thước từ vựng không khớp.")

    # Kiểm tra một vài ID để chắc chắn
    print(f"ID của '<pad>': {final_tokenizer.pad_token_id}")
    print(f"ID của '</s>': {final_tokenizer.eos_token_id}")
    print(f"ID của '<extra_id_0>': {final_tokenizer.convert_tokens_to_ids('<extra_id_0>')}")
    print(f"ID của '<extra_id_99>': {final_tokenizer.convert_tokens_to_ids('<extra_id_99>')}")
    
    print("\nTokenizer đã hoàn toàn sẵn sàng! Hãy dùng thư mục " + f"'{FINAL_HF_TOKENIZER_DIR}'" + " để huấn luyện mô hình.")

except Exception as e:
    print(f"LỖI: Kiểm tra tokenizer thất bại: {e}")

# Dọn dẹp file tạm
print("\n--- Dọn dẹp file tạm ---")
if os.path.exists(f"{MODEL_PREFIX}.model"):
    os.remove(f"{MODEL_PREFIX}.model")
if os.path.exists(f"{MODEL_PREFIX}.vocab"):
    os.remove(f"{MODEL_PREFIX}.vocab")
print("Đã xóa các file tạm.")