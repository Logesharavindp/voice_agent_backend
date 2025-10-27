import json
import difflib
import uuid
import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AzureOpenAI
from gtts import gTTS

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIGURE FASTAPI ----------
app = FastAPI(title="Voice Employment Verification Agent")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIGURE AZURE OPENAI ----------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ---------- LOAD STATIC DATA ----------
with open("static/user.json", "r") as f:
    data = json.load(f)

users = data["users"]
company_list = data["company_list"]

# ---------- SESSION STORAGE ----------
# In production, use Redis or a proper database
sessions = {}

# Track audio files for cleanup
audio_files_to_cleanup = []

# ---------- INDUSTRIAL-STANDARD SYSTEM PROMPT FOR VOICE AGENT ----------
system_prompt = """
    You are a professional Employment Verification Voice Agent conducting structured identity and employment verification calls.

    ## PRIMARY OBJECTIVE
    Collect and verify three critical data points in sequence:
    1. Full Name
    2. Years of Experience (total professional experience)
    3. Date of Birth (MM/DD/YYYY format)
    4. Current Employment Status (company name verification)

    ## CORE BEHAVIORAL RULES

    ### Conversation Flow Management
    - Follow a STRICT LINEAR PROGRESSION: Only move to the next question after successfully collecting the current information.
    - If a user provides multiple pieces of information at once, acknowledge all but process them in order.
    - NEVER skip ahead or ask multiple questions simultaneously.
    - If user provides irrelevant information, politely redirect without answering off-topic questions.

    ### Information Collection Protocols

    **For Name Collection:**
    - Ask: "Hi there! I'm calling to verify your employment. Could I have your full name, please?"
    - If unclear/partial: "Thanks! Could you provide your full first and last name?"
    - If still unclear after 2 attempts: "I want to make sure I have this right. Could you spell your full name for me?"
    - Validation: Must contain at least first and last name.

    **For Years of Experience:**
    - Ask: "Great, [Name]! How many years of professional experience do you have in total?"
    - If non-numeric response: "I need the number of years. For example, 5 years or 10 years. How many years would that be?"
    - If unrealistic (>50 or <0): "Just to confirm, you said [X] years of experience — is that correct?"
    - Validation: Must be a number between 0-50.

    **For Date of Birth:**
    - Ask: "Thanks! And could you provide your date of birth? Month, day, and year please."
    - If incomplete: "I need the complete date. Could you give me the month, day, and year?"
    - If wrong format: "Could you provide that as month, day, year? For example, March 15th, 1990."
    - Validation: Must be a valid date; person must be 18-80 years old.

    **For Employment Verification:**
    - Statement: "According to our records, you're currently employed at [COMPANY_NAME]. Is that correct?"
    - If YES: "Perfect! Your employment has been verified successfully. Thanks for your time, [Name]!"
    - If NO: "No problem, [Name]. Which company are you currently working with?"
    - After they respond: "Let me check... I have a few options that might match: [LIST_OPTIONS]. Does one of these match, or is it a different company?"
    - If they pick one: "Got it — [COMPANY]. Your verification is now complete. Thank you!"
    - If different: "Understood. I've recorded [COMPANY_NAME]. Your verification is complete. Thank you!"

    ## HANDLING DIFFICULT SCENARIOS

    ### User Doesn't Understand the Question
    - Rephrase once using simpler language
    - Example: "Years of experience" → "How long have you been working professionally in total?"
    - If still confused after 2nd attempt: "Let me put it differently: [alternative phrasing]"
    - After 3 failed attempts: "No worries! Let me ask something else and we can come back to this."

    ### User Provides Irrelevant Information or Asks Questions
    Response template: "I appreciate that, but I need to focus on verifying your employment right now. [Restate current question]."

    Examples:
    - User: "What company is this?" 
    You: "I'm with the Employment Verification Department. I need to collect some information first. Could you provide your full name?"
    
    - User: "Why do you need this?"
    You: "This is a standard employment verification call. I'll need to collect a few details to proceed. Let's start with your full name, please."

    - User: "Can I call you back?"
    You: "This will only take about 2 minutes. Let's get through this quickly. What's your full name?"

    ### User Goes Off-Topic During Collection
    - Acknowledge briefly WITHOUT engaging: "I hear you. Right now, I need [current information]. Could you provide that?"
    - If they persist: "[Name], I want to help, but I need to complete this verification first. [Restate question]."
    - NEVER answer unrelated questions until verification is complete.

    ### User Gives Vague/Incomplete Answers
    - "I need a bit more detail. [Specific clarifying question]"
    - Example: User says "Been working a while" → You: "Could you give me the approximate number of years?"

    ### User Refuses to Provide Information
    - First refusal: "I understand. This information is required to complete your employment verification. Could you provide [information] so we can proceed?"
    - Second refusal: "I'm unable to complete the verification without this information. Are you able to provide [information] now?"
    - Third refusal: "I understand this may not be a good time. Unfortunately, I cannot proceed without this information. Is there anything I can help clarify?"

    ### User Provides Information for Wrong Question
    - "Thanks for that information. I'll need that in just a moment. First, could you tell me [current question]?"
    - Store the volunteered information mentally and acknowledge it when you reach that step.

    ## VOICE & TONE GUIDELINES

    **Pacing & Brevity:**
    - Maximum 2-3 sentences per response
    - Pause naturally after questions
    - Speak at a measured, clear pace

    **Emotional Intelligence:**
    - Warm but professional
    - Patient with confusion or hesitation
    - Calm during resistance or frustration
    - Celebratory when verification succeeds

    **Language Style:**
    - Conversational but not casual
    - Avoid jargon (say "date of birth" not "DOB")
    - Use contractions naturally ("I'm" not "I am")
    - Always use the person's name once collected

    **Confidence Markers:**
    - "Perfect," "Got it," "Excellent," "Thanks!"
    - Avoid: "Um," "maybe," "I think," "possibly"

    ## CRITICAL RULES

    1. **NEVER proceed** to the next question without valid data for the current question
    2. **NEVER engage** with off-topic conversations beyond one brief acknowledgment
    3. **ALWAYS redirect** back to the current question within one response
    4. **ALWAYS confirm** information before moving forward ("Just to confirm, [X], correct?")
    5. **NEVER fabricate** or assume information — if unclear, ask again
    6. **LIMIT clarification attempts** to 3 per question, then offer to move forward and return later
    7. **MAINTAIN FOCUS**: Your only job is employment verification — nothing else

    ## SUCCESS CRITERIA
    Verification is complete ONLY when you have:
    ✓ Valid full name
    ✓ Numeric years of experience (0-50)
    ✓ Valid date of birth (age 18-80)
    ✓ Employment status confirmed OR new company name recorded

    End with: "Your verification is complete. Thank you for your time, [Name]!"

    Stay focused, stay professional, and guide every conversation to successful completion.
    """

# ---------- PYDANTIC MODELS ----------
class SessionCreate(BaseModel):
    session_id: Optional[str] = None

class UserMessage(BaseModel):
    session_id: str
    message: str

class AgentResponse(BaseModel):
    session_id: str
    message: str
    audio_url: str
    state: str
    suggestions: Optional[list] = None

# ---------- HELPER FUNCTIONS ----------
def get_employment(email: str):
    """Return employment record if found."""
    return users.get(email)

def suggest_company_matches(user_input: str):
    """Return close company name matches."""
    return difflib.get_close_matches(user_input, company_list, n=5, cutoff=0.4)

def chat_response(conversation_history: list):
    """Use Azure OpenAI to respond conversationally with full conversation history."""
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        messages=messages
    )
    return completion.choices[0].message.content.strip()

def text_to_speech(text: str, session_id: str) -> str:
    """Convert text to speech using gTTS with female voice and return file path."""
    # Create temp audio directory if it doesn't exist
    temp_audio_dir = "temp_audio"
    os.makedirs(temp_audio_dir, exist_ok=True)

    # Generate unique filename in temp directory
    filename = f"{temp_audio_dir}/{session_id}_{uuid.uuid4().hex[:8]}.mp3"

    # Generate speech with gTTS (female voice, slow=False for natural speed)
    tts = gTTS(text=text, lang='en', slow=False, tld='com')
    tts.save(filename)

    # Track file for cleanup
    audio_files_to_cleanup.append(filename)

    return filename

def cleanup_audio_file(filename: str):
    """Delete a temporary audio file."""
    try:
        if os.path.exists(filename):
            os.remove(filename)
            if filename in audio_files_to_cleanup:
                audio_files_to_cleanup.remove(filename)
            print(f"Deleted audio file: {filename}")
    except Exception as e:
        print(f"Error deleting audio file {filename}: {e}")

async def delayed_cleanup_audio(filename: str, delay: int = 30):
    """Delete audio file after a delay (default 30 seconds)."""
    await asyncio.sleep(delay)
    cleanup_audio_file(filename)

def cleanup_old_audio_files():
    """Clean up all tracked audio files."""
    for filename in audio_files_to_cleanup[:]:  # Create a copy to iterate
        cleanup_audio_file(filename)

def save_session_transcript(session_id: str, session_data: dict):
    """Save session transcript to JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Prepare transcript data
    transcript = {
        "session_id": session_id,
        "user_data": session_data.get("user_data", {}),
        "state": session_data.get("state", ""),
        "verified": session_data.get("verified", False),
        "conversation_history": session_data.get("conversation_history", []),
        "timestamp": None  # Will be set when saved
    }

    # Add timestamp
    from datetime import datetime
    transcript["timestamp"] = datetime.now().isoformat()

    # Save to file
    filename = f"output/{session_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    return filename

# ---------- API ENDPOINTS ----------

@app.get("/")
async def root():
    """Serve the HTML interface."""
    return FileResponse("index.html")

@app.post("/api/session/create")
async def create_session(session_data: SessionCreate):
    """Create a new conversation session."""
    session_id = session_data.session_id or str(uuid.uuid4())
    
    sessions[session_id] = {
        "conversation_history": [],
        "state": "GREETING",  # GREETING, COLLECTING_INFO, VERIFYING_EMPLOYMENT, COMPLETED
        "user_data": {},
        "verified": False
    }
    
    # Initial greeting
    greeting = "Hello! Welcome to the Employment Verification System. Let's start by collecting some information. What is your full name?"
    sessions[session_id]["conversation_history"].append({
        "role": "assistant",
        "content": greeting
    })
    
    # Generate audio
    audio_file = text_to_speech(greeting, session_id)
    
    return AgentResponse(
        session_id=session_id,
        message=greeting,
        audio_url=f"/api/audio/{os.path.basename(audio_file)}",
        state="GREETING"
    )

@app.post("/api/chat")
async def chat(user_message: UserMessage):
    """Process user message and return agent response with audio."""
    session_id = user_message.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    user_input = user_message.message.strip()
    
    # Add user message to history
    session["conversation_history"].append({
        "role": "user",
        "content": user_input
    })
    
    response_text = ""
    suggestions = None
    
    # State machine for conversation flow
    if session["state"] == "GREETING":
        # Collect name
        session["user_data"]["name"] = user_input
        first_name = user_input.split()[0]
        session["user_data"]["first_name"] = first_name
        response_text = f"Nice to meet you, {first_name}! How many years of experience do you have?"
        session["state"] = "COLLECTING_EXPERIENCE"
    
    elif session["state"] == "COLLECTING_EXPERIENCE":
        # Collect years of experience
        session["user_data"]["years_of_experience"] = user_input
        response_text = "Great! What is your date of birth? Please say it in a format like day, month, year."
        session["state"] = "COLLECTING_DOB"
    
    elif session["state"] == "COLLECTING_DOB":
        # Collect date of birth
        session["user_data"]["date_of_birth"] = user_input
        first_name = session["user_data"]["first_name"]
        response_text = f"Thank you, {first_name}! Now, please provide your email address so I can check our records."
        session["state"] = "COLLECTING_EMAIL"
    
    elif session["state"] == "COLLECTING_EMAIL":
        # Collect email and check employment
        email = user_input.strip().lower()
        session["user_data"]["email"] = email
        user_data = get_employment(email)
        session["user_data"]["employment_record"] = user_data
        
        first_name = session["user_data"]["first_name"]
        
        if user_data:
            company_name = user_data["company_name"]
            response_text = f"Hi {first_name}, it's great to talk with you! Our records show you work at {company_name}. Is that still correct?"
            session["state"] = "VERIFYING_EMPLOYMENT"
        else:
            response_text = f"I couldn't find any employment records for {email}. Could you please tell me your current company name?"
            session["state"] = "ASKING_COMPANY"
    
    elif session["state"] == "VERIFYING_EMPLOYMENT":
        # Check if user confirms or denies employment
        if user_input.lower() in ["yes", "yeah", "yep", "correct", "right", "that's right", "that's correct", "yes it is"]:
            first_name = session["user_data"]["first_name"]
            response_text = f"Perfect! Thank you {first_name}, your employment verification is complete. Have a great day!"
            session["state"] = "COMPLETED"
            session["verified"] = True
        elif user_input.lower() in ["no", "nope", "not correct", "wrong", "incorrect", "no it's not"]:
            first_name = session["user_data"]["first_name"]
            response_text = f"No worries, {first_name}! Could you please tell me your current company name?"
            session["state"] = "ASKING_COMPANY"
        else:
            # Use AI for natural response
            response_text = chat_response(session["conversation_history"])
    
    elif session["state"] == "ASKING_COMPANY":
        # User provided company name
        company_input = user_input
        matches = suggest_company_matches(company_input)
        
        if matches:
            suggestions = matches
            match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(matches)])
            response_text = f"I found some possible matches: {match_list}. Please say the number of your company, or say your company name if it's not listed."
            session["state"] = "SELECTING_COMPANY"
            session["user_data"]["company_matches"] = matches
        else:
            first_name = session["user_data"]["first_name"]
            response_text = f"Thank you! I've recorded your company as {company_input}. Your verification is complete, {first_name}!"
            session["state"] = "COMPLETED"
            session["verified"] = True
    
    elif session["state"] == "SELECTING_COMPANY":
        # User selects from matches or provides manual entry
        matches = session["user_data"].get("company_matches", [])
        first_name = session["user_data"]["first_name"]
        
        # Check if user said a number
        if user_input.isdigit() and 1 <= int(user_input) <= len(matches):
            selected = matches[int(user_input)-1]
            response_text = f"Great! I've updated your company to {selected}. Thank you {first_name}, your verification is complete!"
            session["state"] = "COMPLETED"
            session["verified"] = True
        else:
            response_text = f"Thank you! I've recorded your company as {user_input}. Your verification is complete, {first_name}!"
            session["state"] = "COMPLETED"
            session["verified"] = True
    
    else:
        # Default: use AI to respond
        response_text = chat_response(session["conversation_history"])
    
    # Add assistant response to history
    session["conversation_history"].append({
        "role": "assistant",
        "content": response_text
    })

    # Save transcript if session is completed
    if session["state"] == "COMPLETED":
        save_session_transcript(session_id, session)

    # Generate audio
    audio_file = text_to_speech(response_text, session_id)

    return AgentResponse(
        session_id=session_id,
        message=response_text,
        audio_url=f"/api/audio/{os.path.basename(audio_file)}",
        state=session["state"],
        suggestions=suggestions
    )

@app.get("/api/audio/{filename}")
async def get_audio(filename: str, background_tasks: BackgroundTasks):
    """Serve audio files and schedule for deletion after 30 seconds."""
    file_path = f"temp_audio/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Schedule cleanup after 30 seconds (enough time for audio to play)
    background_tasks.add_task(delayed_cleanup_audio, file_path, 30)

    return FileResponse(
        file_path,
        media_type="audio/mpeg"
    )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.post("/api/session/{session_id}/save")
async def save_session(session_id: str):
    """Manually save session transcript to JSON file."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    filename = save_session_transcript(session_id, sessions[session_id])
    return {
        "message": "Session transcript saved successfully",
        "filename": filename,
        "session_id": session_id
    }

@app.get("/api/transcripts")
async def list_transcripts():
    """List all saved transcripts."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        return {"transcripts": []}

    transcripts = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcripts.append({
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "verified": data.get("verified"),
                    "user_name": data.get("user_data", {}).get("name"),
                    "filename": filename
                })

    return {"transcripts": transcripts}

@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """Get a specific transcript by session ID."""
    filename = f"output/{session_id}.json"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(filename, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    return transcript

@app.post("/api/cleanup/audio")
async def cleanup_audio():
    """Manually cleanup all temporary audio files."""
    cleanup_old_audio_files()
    return {
        "message": "Audio files cleaned up successfully",
        "remaining_files": len(audio_files_to_cleanup)
    }

@app.delete("/api/audio/{filename}")
async def delete_audio(filename: str):
    """Delete a specific audio file."""
    file_path = f"temp_audio/{filename}"
    cleanup_audio_file(file_path)
    return {"message": f"Audio file {filename} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

