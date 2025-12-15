import google.generativeai as genai
import os
import json
from typing import List, Dict

def ai_analysis(messages: List[Dict], frequency: Dict[str, int]):
    # Set API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        # Fallback for development/testing if needed, or just return mock data
        # For now, we'll proceed and let it fail if key is missing, or return a mock error response
        return {
            "summary": "API Key missing. Please set GEMINI_API_KEY.",
            "advice": ["Check server configuration."],
            "sampleReplies": ["Error: No API Key"]
        }

    genai.configure(api_key=api_key)

    # Configure model for JSON output
    generation_config = {
        "response_mime_type": "application/json",
    }

    # Initialize model
    model = genai.GenerativeModel("gemini-2.5-flash-lite", generation_config=generation_config)

    # Construct prompt
    # We'll format the messages for the prompt
    formatted_messages = json.dumps(messages, ensure_ascii=False, indent=2)
    
    prompt = f"""
    당신은 여성의 마음을 잘 아는 픽업 아티스트입니다.
    이제부터 채팅 대화를 전달할 테니, 요약과 조언, 예시 'USER'입장의 답변을 제공해 주세요.
    ‘OTHER’는 여성, ‘USER’는 남성입니다.
    'USER'의 입장에서 'OTHER'의 마음을 얻기위한 대화를 이어가야 합니다.
    핵심 포인트: 채팅은 만남을 위한 수단일 뿐, 지나친 감정 표현이나 간절해 보이는 태도는 피하세요.
    
    1. 기본 대화 마인드셋 (목적과 태도)
    유일한 목적: 상호 관심 확인과 “만남” 확보를 위한 수단입니다. 긴 대화는 직접 만날 때로 미루세요.
    감정 과잉 피하기: 채팅으로 지나치게 관심을 표현하지 마세요.
    이미지 구축: “우리가 잘 맞는 것 같다”, “만나면 재미있을 것 같다”는 인상만 전달하는 것이 성공입니다.
    가치 증명: 채팅으로 상대방을 평가하세요(예: “운동하세요?”, “요리 잘하세요?”). 자신의 기준이 높음을 암시하세요.
    
    2. 타이밍과 리듬 (밀고 당기는 기술)
    ‘ㅋㅋ’ 타이밍: 무작정 사용하지 마세요. 오직 다음 상황에만 사용하세요:
    1.상대를 놀릴 때
    2.여성이 관심 신호를 보낼 때
    3.여성이 당신에게 좋은 인상을 주려 할 때
    4.분위기를 띄울 때
    답장 속도와 빈도:
    특정 시간대에만 답장하세요.
    낮에는 본업에 집중하며 연락을 자제해 ‘가치 있고 바쁜 남자’임을 묵묵히 증명하라.
    퇴근 후나 밤처럼 그녀의 가장 한가한 시간대를 노려 폭발적인 주고받기 농담으로 ‘보장된 재미’를 선사하라.
    이 리듬 차이는 그녀로 하여금 당신의 메시지를 기대하게 하며 최소한의 시간에 대화 몰입도를 극대화한다.

    3. 스타일과 형식 (시각적 표현)
    간결하게: 한 번에 한 줄만 보냅니다. 긴 문자는 금물.
    이모지: 과도하게 사용하지 마세요. 적절한 순간에만 사용해 세심함을 보여주세요.
    어조와 태도: 유쾌함과 성숙함의 균형을 유지하세요.
    부담을 줄이세요: “제발 만나주세요” 대신 “만날 기회를 드리겠습니다”라고 전달해 무거운 일회성 압박을 피하세요.
    선택권을 주세요: 만남을 전제로 특별한 특권을 부여하는 듯한 느낌으로 대화를 구성하세요.

    4. 주의사항
    급하게 만나고 싶어 안달하는 모습을 보이지 마세요.
    채팅으로 감정을 고백하거나 관계를 굳히려 하지 마세요.
    만남은 어느정도 대화의 친밀감이 생겼을때만 진행하세요.
    대화 시 적절한 예의를 지키세요.


    Message Frequency: {json.dumps(frequency, ensure_ascii=False)}
    
    Conversation History:
    {formatted_messages}
    
    다음 JSON 형식으로 출력해 주세요. 값은 반드시 한국어로 표기해야 합니다:
    {{
        "summary": "A brief summary of the conversation content and relationship dynamics in Korean.",
        "advice": ["Tip 1", "Tip 2", "Tip 3"],
        "sampleReplies": ["Reply option 1", "Reply option 2", "Reply option 3"] The key point is to carefully consider what the USER's next response should be.
    }}
    """

    try:
        # API call
        response = model.generate_content(prompt)
        # --- 1. 토큰 사용량 계산 및 출력 ---
        usage_metadata = response.usage_metadata
        if usage_metadata:
            input_tokens = usage_metadata.prompt_token_count
            output_tokens = usage_metadata.candidates_token_count
            total_tokens = usage_metadata.total_token_count

            print("\n" + "="*40)
            print("✨ Gemini API Token Usage Details ✨")
            print(f"  Input Tokens (Prompt):  {input_tokens} tokens")
            print(f"  Output Tokens (Response): {output_tokens} tokens")
            print(f"  Total Tokens Used:      {total_tokens} tokens")
            print("="*40 + "\n")
        # -----------------------------------
        # Convert result to JSON
        result_data = json.loads(response.text)

        # Output AI result
        print("### AI Decision: ", "###")
        print(json.dumps(result_data, ensure_ascii=False, indent=2))
    
        return result_data
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "summary": "Failed to analyze conversation due to an error.",
            "advice": ["Please try again later."],
            "sampleReplies": ["Error occurred"]
        }
