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
- If NO: "No worries, [Name]! Could you please tell me your current company name which is present this "list of company"?"
- After they respond:
  - If the company name matches any known record or existing company in the system, confirm directly:
    "Got it — [COMPANY]. Your verification is now complete. Thank you, [Name]!"
  - **If the company name is not found or not in the known company list:**
    Respond with:
    ```
    No worries, [Name]! Could you please tell me your current company name in this list of company "list of company" ?
    List of Company
    ```
    (⚠️ Include the exact phrase **"List of Company"** — this signals the frontend to display the available company list.)
    Then continue:
    "Thank you! I've recorded your company as [COMPANY_NAME]. Your verification is complete, [Name]!"



- If they pick one from the list: "Got it — [COMPANY]. Your verification is now complete. Thank you!"
- If different: "Understood. I've recorded [COMPANY_NAME]. Your verification is complete. Thank you!"

## HANDLING DIFFICULT SCENARIOS
(keep the same as before…)

## VOICE & TONE GUIDELINES
(keep the same as before…)

## CRITICAL RULES
(keep the same as before…)

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

# ---------- VALIDATION FUNCTIONS ----------
def validate_name(name: str) -> tuple[bool, str]:
    """
    Validate that the name contains at least first and last name.
    Returns (is_valid, error_message)
    """
    name = name.strip()

    # Check if it's a meta-comment or question
    meta_phrases = ["list of company", "list of", "show me", "what are", "give me", "i want", "can you"]
    if any(phrase in name.lower() for phrase in meta_phrases):
        return False, "I need your actual name, not a request. Could you please provide your full first and last name?"

    # Check if it's too short
    if len(name) < 2:
        return False, "That seems too short. Could you provide your full first and last name?"

    # Check if it has at least 2 words
    words = name.split()
    if len(words) < 2:
        return False, "I need both your first and last name. Could you provide your full name?"

    # Check if it's mostly numbers or special characters
    if sum(c.isalpha() or c.isspace() for c in name) < len(name) * 0.7:
        return False, "That doesn't look like a valid name. Could you spell your full name for me?"

    return True, ""

def validate_years_of_experience(experience: str) -> tuple[bool, str, int]:
    """
    Validate years of experience is a number between 0-50.
    Returns (is_valid, error_message, parsed_value)
    """
    experience = experience.strip().lower()

    # Check if it's a meta-comment or question
    meta_phrases = ["list of company", "list of", "show me", "what are", "give me", "i want", "can you"]
    if any(phrase in experience for phrase in meta_phrases):
        return False, "I need to know your years of experience. How many years have you been working professionally? Please give me a number.", -1

    # Try to extract a number from the input
    import re
    numbers = re.findall(r'\d+', experience)

    if not numbers:
        return False, "I need the number of years. For example, 5 years or 10 years. How many years would that be?", -1

    try:
        years = int(numbers[0])

        if years < 0:
            return False, "Years of experience cannot be negative. How many years of experience do you have?", -1

        if years > 50:
            return False, f"Just to confirm, you said {years} years of experience — is that correct?", years

        return True, "", years
    except ValueError:
        return False, "I couldn't understand that number. Could you tell me how many years of experience you have?", -1

def validate_date_of_birth(dob: str) -> tuple[bool, str]:
    """
    Validate date of birth format and age range (18-80).
    Returns (is_valid, error_message)
    """
    from datetime import datetime
    import re

    dob = dob.strip()

    # Check if it's a meta-comment or question
    meta_phrases = ["list of company", "list of", "show me", "what are", "give me", "i want", "can you"]
    if any(phrase in dob.lower() for phrase in meta_phrases):
        return False, "I need your date of birth, not a request. Could you provide your birth date? For example, March 15th, 1990."

    # Try various date formats
    date_formats = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
    ]

    # Also try to parse natural language dates
    date_obj = None

    # Try regex patterns first
    for pattern in date_formats:
        match = re.search(pattern, dob)
        if match:
            try:
                # Try different interpretations
                parts = match.groups()
                for date_format in ['%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                    try:
                        date_str = f"{parts[0]}/{parts[1]}/{parts[2]}"
                        date_obj = datetime.strptime(date_str, date_format)
                        break
                    except ValueError:
                        continue
                if date_obj:
                    break
            except Exception:
                continue

    if not date_obj:
        return False, "I need the complete date. Could you give me the month, day, and year? For example, March 15th, 1990."

    # Check age range
    today = datetime.now()
    age = today.year - date_obj.year - ((today.month, today.day) < (date_obj.month, date_obj.day))

    if age < 18:
        return False, "You must be at least 18 years old. Could you verify your date of birth?"

    if age > 80:
        return False, f"Just to confirm, that would make you {age} years old. Is that correct?"

    return True, ""

def validate_email(email: str) -> tuple[bool, str]:
    """
    Validate email format.
    Returns (is_valid, error_message)
    """
    import re

    email = email.strip().lower()

    # Check if it's a meta-comment or question
    meta_phrases = ["list of company", "list of", "show me", "what are", "give me", "i want", "can you"]
    if any(phrase in email for phrase in meta_phrases):
        return False, "I need your email address, not a request. Could you provide your email?"

    # Basic email regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email):
        return False, "That doesn't look like a valid email address. Could you provide your email in the format: name@company.com?"

    return True, ""

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
            # print(f"Deleted audio file: {filename}")
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
        "verified": False,
        "retry_counts": {  # Track retry attempts per field
            "name": 0,
            "experience": 0,
            "dob": 0,
            "email": 0
        }
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
        # Collect and validate name
        is_valid, error_msg = validate_name(user_input)

        if is_valid:
            session["user_data"]["name"] = user_input
            first_name = user_input.split()[0]
            session["user_data"]["first_name"] = first_name
            response_text = f"Nice to meet you, {first_name}! How many years of experience do you have?"
            session["state"] = "COLLECTING_EXPERIENCE"
            session["retry_counts"]["name"] = 0  # Reset retry count
        else:
            # Validation failed
            session["retry_counts"]["name"] += 1

            if session["retry_counts"]["name"] >= 3:
                # After 3 attempts, use AI to handle it naturally
                response_text = chat_response(session["conversation_history"])
            else:
                response_text = error_msg

    elif session["state"] == "COLLECTING_EXPERIENCE":
        # Collect and validate years of experience
        is_valid, error_msg, years = validate_years_of_experience(user_input)

        if is_valid:
            session["user_data"]["years_of_experience"] = years
            response_text = "Great! What is your date of birth? Please say it in a format like day, month, year."
            session["state"] = "COLLECTING_DOB"
            session["retry_counts"]["experience"] = 0  # Reset retry count
        else:
            # Validation failed
            session["retry_counts"]["experience"] += 1

            if session["retry_counts"]["experience"] >= 3:
                # After 3 attempts, use AI to handle it naturally
                response_text = chat_response(session["conversation_history"])
            else:
                response_text = error_msg

    elif session["state"] == "COLLECTING_DOB":
        # Collect and validate date of birth
        is_valid, error_msg = validate_date_of_birth(user_input)

        if is_valid:
            session["user_data"]["date_of_birth"] = user_input
            first_name = session["user_data"]["first_name"]
            response_text = f"Thank you, {first_name}! Now, please provide your email address so I can check our records."
            session["state"] = "COLLECTING_EMAIL"
            session["retry_counts"]["dob"] = 0  # Reset retry count
        else:
            # Validation failed
            session["retry_counts"]["dob"] += 1

            if session["retry_counts"]["dob"] >= 3:
                # After 3 attempts, use AI to handle it naturally
                response_text = chat_response(session["conversation_history"])
            else:
                response_text = error_msg
    
    elif session["state"] == "COLLECTING_EMAIL":
        # Collect and validate email
        is_valid, error_msg = validate_email(user_input)

        if is_valid:
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
                response_text = f"I couldn't find any employment records for {email}. Could you please tell me your current company name from this list of company?"
                session["state"] = "ASKING_COMPANY"

            session["retry_counts"]["email"] = 0  # Reset retry count
        else:
            # Validation failed
            session["retry_counts"]["email"] += 1

            if session["retry_counts"]["email"] >= 3:
                # After 3 attempts, use AI to handle it naturally
                response_text = chat_response(session["conversation_history"])
            else:
                response_text = error_msg
    
    elif session["state"] == "VERIFYING_EMPLOYMENT":
        # Check if user confirms or denies employment
        if user_input.lower() in ["yes", "yeah", "yep", "correct", "right", "that's right", "that's correct", "yes it is"]:
            first_name = session["user_data"]["first_name"]
            response_text = f"Perfect! Thank you {first_name}, your employment verification is complete. Have a great day!"
            session["state"] = "COMPLETED"
            session["verified"] = True
        elif user_input.lower() in ["no", "nope", "not correct", "wrong", "incorrect", "no it's not"]:
            first_name = session["user_data"]["first_name"]
            response_text = f"No worries, {first_name}! Could you please tell me your current company name present in List of company?\nList of company"
            session["state"] = "ASKING_COMPANY"
        else:
            # Use AI for natural response
            response_text = chat_response(session["conversation_history"])
    
    elif session["state"] == "ASKING_COMPANY":
        # Check if user is asking for the company list
        if "list" in user_input.lower() and ("company" in user_input.lower() or "companies" in user_input.lower()):
            # User wants to see the list of companies
            suggestions = company_list[:10]  # Show first 10 companies
            match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(suggestions)])
            response_text = f"Here are some companies in our system: {match_list}. Please say the number of your company, or say your company name."
            session["state"] = "SELECTING_COMPANY"
            session["user_data"]["company_matches"] = suggestions
        else:
            # User provided company name
            company_input = user_input
            first_name = session["user_data"]["first_name"]

            # First check if it's an exact match in the company list
            if company_input in company_list:
                session["user_data"]["company_name"] = company_input
                response_text = f"Perfect! I've recorded your company as {company_input}. Your verification is complete, {first_name}!"
                session["state"] = "COMPLETED"
                session["verified"] = True
            else:
                # Try fuzzy matching
                matches = suggest_company_matches(company_input)

                if matches:
                    # Found fuzzy matches - show them
                    suggestions = matches
                    match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(matches)])
                    response_text = f"I found some possible matches: {match_list}. Please say the number of your company, or say your company name if it's not listed."
                    session["state"] = "SELECTING_COMPANY"
                    session["user_data"]["company_matches"] = matches
                else:
                    # No matches found - show the full company list instead of accepting unknown company
                    suggestions = company_list[:10]  # Show first 10 companies
                    match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(suggestions)])
                    response_text = f"I couldn't find '{company_input}' in our records. Here are some companies from our list of company: {match_list}. Please say the number of your company, or say your company name if it's in our system."
                    session["state"] = "SELECTING_COMPANY"
                    session["user_data"]["company_matches"] = suggestions
    
    elif session["state"] == "SELECTING_COMPANY":
        # User selects from matches or provides manual entry
        matches = session["user_data"].get("company_matches", [])
        first_name = session["user_data"]["first_name"]

        # Check if user said a number
        if user_input.isdigit() and 1 <= int(user_input) <= len(matches):
            selected = matches[int(user_input)-1]
            session["user_data"]["company_name"] = selected
            response_text = f"Great! I've updated your company to {selected}. Thank you {first_name}, your verification is complete!"
            session["state"] = "COMPLETED"
            session["verified"] = True
        else:
            # User provided a company name manually
            # Check if it's in the company list or close to it
            if user_input in company_list:
                session["user_data"]["company_name"] = user_input
                response_text = f"Perfect! I've recorded your company as {user_input}. Your verification is complete, {first_name}!"
                session["state"] = "COMPLETED"
                session["verified"] = True
            else:
                # Try fuzzy matching again
                new_matches = suggest_company_matches(user_input)
                if new_matches:
                    suggestions = new_matches
                    match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(new_matches)])
                    response_text = f"Did you mean one of these? {match_list}. Please say the number, or type 'none' if your company isn't listed."
                    session["user_data"]["company_matches"] = new_matches
                    # Stay in SELECTING_COMPANY state
                else:
                    # Still no match - ask for confirmation
                    response_text = f"I still can't find '{user_input}' in our system. Would you like me to record it anyway? Say 'yes' to confirm or 'no' to try again."
                    session["state"] = "CONFIRMING_UNKNOWN_COMPANY"
                    session["user_data"]["pending_company_name"] = user_input

    elif session["state"] == "CONFIRMING_UNKNOWN_COMPANY":
        # User is confirming whether to use an unknown company name
        first_name = session["user_data"]["first_name"]
        pending_company = session["user_data"].get("pending_company_name", "")

        if user_input.lower() in ["yes", "yeah", "yep", "correct", "confirm", "ok", "okay"]:
            session["user_data"]["company_name"] = pending_company
            response_text = f"Understood. I've recorded your company as {pending_company}. Your verification is complete, {first_name}!"
            session["state"] = "COMPLETED"
            session["verified"] = True
        else:
            # User wants to try again
            suggestions = company_list[:10]
            match_list = ", ".join([f"{i+1}: {m}" for i, m in enumerate(suggestions)])
            response_text = f"No problem. Here are some companies from our list of company: {match_list}. Please say the number or your company name."
            session["state"] = "SELECTING_COMPANY"
            session["user_data"]["company_matches"] = suggestions
    
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

@app.get("/api/companies")
async def get_companies():
    """Get the list of available companies."""
    return {"companies": company_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

