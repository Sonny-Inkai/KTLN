from generate import generate_relations

input_text = """
Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở. 3. Việc phối hợp phải thường xuyên, kịp thời, đồng bộ, chặt chẽ, thống nhất, đúng quy định của pháp luật.
"""
result = generate_relations(None, None, None, input_text, max_length=512)
print(result)









