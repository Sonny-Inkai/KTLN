import os
from huggingface_hub import HfApi, HfFolder, login, create_repo
from transformers import AutoTokenizer

def upload_tokenizer_to_hf_hub(local_tokenizer_dir: str, repo_id: str, hf_token: str = None, is_private: bool = False):
    """
    Tải tokenizer từ thư mục cục bộ và đăng lên Hugging Face Hub.

    Args:
        local_tokenizer_dir (str): Đường dẫn đến thư mục chứa tokenizer đã huấn luyện cục bộ.
        repo_id (str): ID của repository trên Hugging Face Hub (ví dụ: "username/tokenizer-name").
        hf_token (str, optional): Token API của Hugging Face. Nếu là None, sẽ cố gắng sử dụng token đã lưu
                                   hoặc nhắc người dùng đăng nhập.
        is_private (bool, optional): Đặt repository là private hay public. Mặc định là False (public).
    """
    print(f"Bắt đầu quá trình đăng tokenizer từ '{local_tokenizer_dir}' lên '{repo_id}'...")

    # --- 1. Kiểm tra thư mục tokenizer cục bộ ---
    if not os.path.isdir(local_tokenizer_dir):
        print(f"Lỗi: Không tìm thấy thư mục tokenizer cục bộ tại '{local_tokenizer_dir}'.")
        print("Vui lòng kiểm tra lại đường dẫn.")
        return

    # --- 2. Đăng nhập vào Hugging Face Hub (nếu cần) ---
    # Ưu tiên token được truyền vào, sau đó là token đã lưu, cuối cùng là nhắc đăng nhập.
    token_to_use = hf_token
    if not token_to_use:
        token_to_use = HfFolder.get_token() # Lấy token đã lưu nếu có

    if not token_to_use:
        print("\nKhông tìm thấy token Hugging Face đã lưu.")
        print("Đang cố gắng đăng nhập. Vui lòng làm theo hướng dẫn (có thể mở trình duyệt hoặc yêu cầu nhập token).")
        try:
            login() # Sẽ nhắc người dùng cung cấp token nếu chưa đăng nhập
            token_to_use = HfFolder.get_token()
            if not token_to_use:
                print("Đăng nhập không thành công hoặc không lấy được token. Vui lòng thử lại.")
                return
            print("Đăng nhập thành công!")
        except Exception as e:
            print(f"Lỗi trong quá trình đăng nhập: {e}")
            print("Hãy thử đăng nhập thủ công bằng 'huggingface-cli login' trong terminal rồi chạy lại script.")
            return
    else:
        print("Đã tìm thấy token Hugging Face.")

    # --- 3. Tạo repository trên Hugging Face Hub (nếu chưa có) ---
    # Sử dụng HfApi để có nhiều quyền kiểm soát hơn
    api = HfApi(token=token_to_use)
    try:
        # Kiểm tra xem repo đã tồn tại chưa
        try:
            api.repo_info(repo_id=repo_id)
            print(f"Repository '{repo_id}' đã tồn tại trên Hugging Face Hub.")
        except Exception: # RepoResourceNotFound or other errors
            print(f"Repository '{repo_id}' chưa tồn tại. Đang tạo mới...")
            create_repo(repo_id, token=token_to_use, private=is_private, repo_type="model", exist_ok=True)
            print(f"Đã tạo thành công repository '{repo_id}'.")
    except Exception as e:
        print(f"Lỗi khi tạo hoặc kiểm tra repository '{repo_id}': {e}")
        return

    # --- 4. Tải tokenizer cục bộ ---
    print(f"\nĐang tải tokenizer từ thư mục cục bộ: '{local_tokenizer_dir}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_dir)
        print("Tải tokenizer cục bộ thành công.")
    except Exception as e:
        print(f"Lỗi khi tải tokenizer từ '{local_tokenizer_dir}': {e}")
        print("Hãy đảm bảo thư mục chứa các file tokenizer hợp lệ (tokenizer_config.json, .model, special_tokens_map.json).")
        return

    # --- 5. Đẩy tokenizer lên Hugging Face Hub ---
    print(f"\nĐang đẩy tokenizer lên repository '{repo_id}'...")
    try:
        # push_to_hub sẽ tự động tải lên các file cần thiết từ thư mục tokenizer
        # bao gồm cả file .model của SentencePiece và các file JSON cấu hình.
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message="Upload tokenizer files",
            private=is_private,
            token=token_to_use, # Đảm bảo sử dụng token đã xác thực
            # create_repo=True # Đã xử lý ở trên, nhưng để đây cũng không sao
        )
        hub_url = f"https://huggingface.co/{repo_id}"
        print("\n--------------------------------------------------------------------")
        print("🎉 ĐĂNG TOKENIZER LÊN HUGGING FACE HUB THÀNH CÔNG! 🎉")
        print("--------------------------------------------------------------------")
        print(f"Cháu có thể xem tokenizer của mình tại: {hub_url}")
        print("\nĐể sử dụng tokenizer này trong các dự án khác, hãy dùng:")
        print("from transformers import AutoTokenizer")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")
        print("--------------------------------------------------------------------")

    except Exception as e:
        print(f"Lỗi khi đẩy tokenizer lên Hugging Face Hub: {e}")
        print("Một số nguyên nhân phổ biến:")
        print("- Token không có quyền 'write'.")
        print("- Vấn đề về kết nối mạng.")
        print("- Repository có thể đã có file với tên tương tự và gây xung đột (ít gặp với tokenizer).")

if __name__ == "__main__":
    print("--- SCRIPT ĐĂNG TOKENIZER LÊN HUGGING FACE HUB ---")

    # Lấy thông tin từ người dùng
    default_local_dir = "./vietnamese_legal_hf_tokenizer"
    local_dir_input = input(f"Nhập đường dẫn đến thư mục tokenizer cục bộ của cháu (mặc định: {default_local_dir}): ")
    local_tokenizer_dir = local_dir_input if local_dir_input else default_local_dir

    # Hướng dẫn về repo_id
    print("\nLưu ý về 'repo_id':")
    print("Đây là tên định danh cho tokenizer của cháu trên Hugging Face Hub.")
    print("Nó thường có dạng: 'ten_username_hf_cua_chau/ten_tokenizer_mong_muon'")
    print("Ví dụ: 'john_doe/vietnamese-legal-tokenizer'")
    repo_id_input = input("Nhập repo_id cháu muốn sử dụng trên Hugging Face Hub: ")
    if not repo_id_input:
        print("Lỗi: repo_id không được để trống.")
        exit()

    private_input = input("Cháu có muốn đặt tokenizer này ở chế độ riêng tư (private) không? (yes/no, mặc định: no): ").lower()
    is_private_repo = private_input == 'yes'

    # Không yêu cầu token trực tiếp trong script để tăng tính bảo mật.
    # Khuyến khích người dùng đăng nhập qua `huggingface-cli login` trước.
    # Hoặc hàm login() sẽ tự xử lý.
    print("\nScript sẽ cố gắng sử dụng token Hugging Face đã được lưu trên máy của cháu.")
    print("Nếu chưa đăng nhập, một lời nhắc (có thể mở trình duyệt) sẽ xuất hiện để cháu cung cấp token.")

    upload_tokenizer_to_hf_hub(local_tokenizer_dir, repo_id_input, is_private=is_private_repo)
