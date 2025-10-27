# Voice Agent Backend

Employment Verification Voice Agent - A conversational AI system for verifying employment information using Azure OpenAI and text-to-speech.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- pip (Python package manager)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd voice_agent_backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the project root with the following variables:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   ```

   Replace the values with your actual Azure OpenAI credentials.

## Running the Application

### Option 1: Run the FastAPI Server (Recommended)

Start the web server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

**Available Endpoints:**
- `GET /` - Main interface
- `POST /api/session/create` - Create a new conversation session
- `POST /api/chat` - Send messages to the agent
- `GET /api/audio/{filename}` - Get generated audio files
- `GET /api/transcripts` - List all saved transcripts
- `GET /api/transcript/{session_id}` - Get specific transcript

### Option 2: Run the Prototype CLI Version

For a command-line interface version:
```bash
python voice_agent_prototype.py
```

This will start an interactive console-based employment verification chat.

## Project Structure

```
voice_agent_backend/
├── main.py                    # FastAPI web server
├── voice_agent_prototype.py   # CLI prototype version
├── requirements.txt           # Python dependencies
├── static/
│   └── user.json             # User employment data
├── output/                    # Saved conversation transcripts
└── temp_audio/               # Temporary audio files (auto-created)
```

## Usage

1. Start the server using `python main.py`
2. Create a session via `/api/session/create`
3. The agent will greet you and ask for:
   - Full name
   - Years of experience
   - Date of birth
   - Email address
4. The agent will verify employment information and provide audio responses
5. Transcripts are automatically saved to the `output/` directory

## Features

- ✅ Conversational AI using Azure OpenAI
- ✅ Text-to-speech audio generation
- ✅ Employment verification against database
- ✅ Company name fuzzy matching
- ✅ Session management
- ✅ Transcript saving
- ✅ RESTful API endpoints

## Troubleshooting

**Issue: Module not found errors**
- Solution: Run `pip install -r requirements.txt`

**Issue: Azure OpenAI authentication errors**
- Solution: Check your `.env` file has correct credentials

**Issue: Audio files not generating**
- Solution: Ensure `temp_audio/` directory exists (created automatically)

## Development

To run with auto-reload during development:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
