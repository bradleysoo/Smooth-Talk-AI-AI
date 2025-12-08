from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import json
from ai_service import ai_analysis

app = FastAPI()

class AnalysisMessageDto(BaseModel):
    sender: str
    text: str
    timeLabel: Optional[str] = None

class AnalysisRequest(BaseModel):
    messages: List[AnalysisMessageDto]

class AnalysisResponse(BaseModel):
    conversationId: Optional[int] = None
    summary: str
    advice: List[str]
    sampleReplies: List[str]
    messageFrequency: Dict[str, int]
    timeFrequency: Dict[str, int]  # Hour -> Count

def calculate_time_frequency(messages: List[AnalysisMessageDto]) -> Dict[str, int]:
    """Calculate hourly message frequency for OTHER sender."""
    time_freq = {}
    
    for msg in messages:
        if msg.sender == "OTHER" and msg.timeLabel:
            try:
                # Parse "YYYY-MM-DD HH:mm" format
                time_part = msg.timeLabel.split(" ")
                if len(time_part) >= 2:
                    hour = time_part[1].split(":")[0]  # Extract hour
                    time_freq[hour] = time_freq.get(hour, 0) + 1
            except Exception as e:
                print(f"Error parsing timeLabel: {msg.timeLabel}, error: {e}")
                continue
    
    return time_freq

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    print(f"Received request with {len(request.messages)} messages")
    
    # Calculate message frequency by sender
    frequency = {}
    for msg in request.messages:
        sender = msg.sender
        frequency[sender] = frequency.get(sender, 0) + 1
    
    print(f"Message Frequency: {frequency}")
    
    # Calculate time frequency
    time_frequency = calculate_time_frequency(request.messages)
    print(f"Time Frequency: {time_frequency}")
    
    # Call AI Service
    messages_dict = [msg.model_dump() for msg in request.messages]
    print("Parsed messages data:")
    print(json.dumps(messages_dict, indent=2, ensure_ascii=False))
    ai_result = ai_analysis(messages_dict, frequency)
    
    return AnalysisResponse(
        conversationId=None,
        summary=ai_result.get("summary", "Analysis failed"),
        advice=ai_result.get("advice", []),
        sampleReplies=ai_result.get("sampleReplies", []),
        messageFrequency=frequency,
        timeFrequency=time_frequency
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
