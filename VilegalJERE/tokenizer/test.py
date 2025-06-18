from transformers import AutoTokenizer
import os

# Thư mục chứa tokenizer đã huấn luyện
TOKENIZER_DIR = "./vietnamese_legal_hf_tokenizer"

def test_tokenizer():
    if not os.path.exists(TOKENIZER_DIR):
        print(f"Lỗi: Thư mục tokenizer '{TOKENIZER_DIR}' không tồn tại.")
        print("Vui lòng chạy script 'train_tokenizer.py' trước.")
        return

    print(f"Đang tải tokenizer từ: {TOKENIZER_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        print("Tải tokenizer thành công!")
    except Exception as e:
        print(f"Lỗi khi tải tokenizer: {e}")
        return

    # --- Kiểm tra các thông tin cơ bản ---
    print(f"\nKích thước từ vựng (Vocab size): {tokenizer.vocab_size}")
    print(f"Token UNK: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"Token BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"Token EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"Token PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # --- Kiểm tra các special tokens đã thêm ---
    print("\nKiểm tra các special tokens nghiệp vụ:")
    domain_tokens_to_test = ["<ORGANIZATION>", "<LEGAL_PROVISION>", "<Effective_From>"]
    for token_str in domain_tokens_to_test:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"Token: '{token_str}', ID: {token_id}")
        if token_id == tokenizer.unk_token_id:
            print(f"  CẢNH BÁO: Token '{token_str}' không được nhận diện đúng, trả về UNK ID!")

    print("\nKiểm tra một vài <extra_id_X> tokens:")
    extra_id_tokens_to_test = ["<extra_id_0>", "<extra_id_50>", "<extra_id_99>"]
    for token_str in extra_id_tokens_to_test:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"Token: '{token_str}', ID: {token_id}")
        if token_id == tokenizer.unk_token_id:
             print(f"  CẢNH BÁO: Token '{token_str}' không được nhận diện đúng, trả về UNK ID!")


    # --- Kiểm tra mã hóa và giải mã ---
    print("\n--- Kiểm tra mã hóa (Encoding) ---")
    sentences = [
        "Điều 1: Đây là một văn bản luật.",
        "Chính phủ ban hành <LEGAL_PROVISION> số 01/1999/NĐ-CP.",
        "Hiệu lực từ <DATE/TIME> <extra_id_5>.",
        "<ORGANIZATION> ABC phải tuân thủ."
    ]

    for i, sentence in enumerate(sentences):
        print(f"\nCâu {i+1}: \"{sentence}\"")

        # Mã hóa
        encoded_output = tokenizer(sentence)
        input_ids = encoded_output.input_ids
        attention_mask = encoded_output.attention_mask # Thường không quan trọng với T5 encoder

        print(f"  Input IDs: {input_ids}")
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"  Tokens: {tokens}")

        # Giải mã
        decoded_sentence = tokenizer.decode(input_ids, skip_special_tokens=False)
        # skip_special_tokens=True để loại bỏ <s>, </s>, <pad> nếu không muốn thấy
        decoded_sentence_skip_special = tokenizer.decode(input_ids, skip_special_tokens=True)

        print(f"  Decoded (raw): \"{decoded_sentence}\"")
        print(f"  Decoded (skip special): \"{decoded_sentence_skip_special}\"")

    # --- Kiểm tra padding ---
    print("\n--- Kiểm tra Padding ---")
    batch_sentences = [
        "Câu ngắn.",
        "Đây là một câu dài hơn một chút."
    ]
    encoded_batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    print("Encoded batch input_ids:")
    print(encoded_batch.input_ids)
    print("Encoded batch attention_mask:")
    print(encoded_batch.attention_mask)
    print("Decoded batch (từng câu):")
    for ids in encoded_batch.input_ids:
        print(f"  - \"{tokenizer.decode(ids, skip_special_tokens=True)}\"")

if __name__ == "__main__":
    test_tokenizer()