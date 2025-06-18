# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types

# --- ENTITY AND RELATION DEFINITIONS ---
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

def extract_legal_relations(input_text):
    """
    Extract legal entity relationships from Vietnamese text using Gemini 2.5 Flash
    
    Args:
        input_text (str): The Vietnamese legal text to analyze
    
    Returns:
        str: Extracted relations in the specified format
    """
    client = genai.Client(
        api_key="AIzaSyC7S-71uyd628QEq2IvMCL8RCOwMyEDtkk",
    )

    # Format the prompt with the input text
    formatted_prompt = SELF_VERIFY_PROMPT_TEMPLATE.format(
        entity_definitions=ENTITY_DEFINITIONS,
        relation_definitions=RELATION_DEFINITIONS,
        item_id="text",
        text_context=input_text
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=formatted_prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        temperature=0.0,
        response_mime_type="text/plain",
    )

    try:
        result_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result_text += chunk.text
        
        return result_text.strip()
    
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def generate():
    """
    Example usage of the extract_legal_relations function
    """
    # Example Vietnamese legal text
    sample_text = """
    Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở. 3. Việc phối hợp phải thường xuyên, kịp thời, đồng bộ, chặt chẽ, thống nhất, đúng quy định của pháp luật.
    """
    
    result = extract_legal_relations(sample_text)
    if result:
        print("Extracted Relations:")
        print(result)
    else:
        print("No relations found or error occurred")

if __name__ == "__main__":
    generate() 