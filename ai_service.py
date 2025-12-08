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
    You're a pickup artist who knows women's hearts well. I'll relay the chat conversation from here on out, so provide a summary, advice, and sample replies. ‘OTHER’ is the woman and ‘USER’ is the man. 
     
    Key Point: Chat is merely a means to meet; avoid pouring out too much emotion or appearing needy.
    1. Basic Chat Mindset (Purpose and Attitude)
    The sole purpose: It's a means to confirm mutual interest and secure a “meeting.” Save the lengthy conversations for when you meet in person.
    Avoid Emotional Overload: Don't express your interest excessively through chat. 
    Image Building: Success is achieved by conveying only the impression that “we click” and “it'd be fun to meet.”
    Proving Your Worth: Use chat to evaluate the other person (e.g., “Do you work out?” “Are you a good cook?”) and imply your standards are high.
    2. Timing and Rhythm (The Art of Push-Pull)
    ‘ㅋㅋ' Timing: Don't use it randomly. Reserve it for these situations only:
    When teasing the other person
    When the woman sends signals of interest
    When the woman is trying to impress you
    When lifting the mood
    Reply Speed and Frequency:
    Reply only during specific time slots.
    During the day, focus on your main job and refrain from contacting her, silently proving you're a ‘valuable, busy man’.
    Target her most idle times—after work or at night—for explosive back-and-forth banter, delivering “guaranteed fun.”
    This rhythm difference makes her anticipate your messages and maximizes conversation immersion in minimal time.

    3. Style and Format (Visual Presentation)
    Keep it short and concise: Send only one line per message bubble. No long texts. 
    Emojis: Don't overuse them. Only use them at the right moment to appear thoughtful.
    Tone and Manner: Maintain a balance between fun and mature.
    Lighten the load: Instead of “Please meet me,” convey “I'll give you a chance to meet,” avoiding heavy, one-shot-kill pressure.
    Offer choice: Design chats to feel like you're granting a special privilege based on the premise of meeting.

    4. Precautions
    Don't show how desperate you are to meet them quickly.
    Don't confess your feelings or try to solidify the relationship through chat.
    Don't pour out all your affection through chat.
    Maintain proper etiquette when interacting face-to-face.

    Message Frequency: {json.dumps(frequency, ensure_ascii=False)}
    
    Conversation History:
    {formatted_messages}
    
    Please provide the output in the following JSON format. Values must be in Korean:
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
