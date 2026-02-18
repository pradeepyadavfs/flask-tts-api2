# Code Documentation

This document explains the two main applications in this repository:

---

## 1. Pharmacy AI Agent (main.py, pharmacy_functions.py, config.json)

### Purpose
A voice-based pharmacy assistant that uses AI to help customers with:
- Getting drug information
- Placing prescription orders
- Looking up existing orders

### How It Works

#### Architecture
```
Twilio Phone Call → WebSocket Server → Deepgram STT → OpenAI GPT-4o-mini → Deepgram TTS → Twilio
```

1. **Twilio Connection**: The server listens on port 5000 for incoming Voice calls via WebSocket
2. **Speech-to-Text**: Uses Deepgram's nova-3 model to convert voice to text
3. **AI Processing**: OpenAI GPT-4o-mini processes the conversation and decides when to call functions
4. **Functions**: Three pharmacy functions are available:
   - `get_drug_info(drug_name)` - Get drug information
   - `place_order(customer_name, drug_name)` - Place a new order
   - `lookup_order(order_id)` - Look up existing order
5. **Text-to-Speech**: Uses Deepgram's aura-2-thalia-en for voice output

#### Key Files

| File | Description |
|------|-------------|
| `main.py` | Main WebSocket server handling Twilio connections |
| `pharmacy_functions.py` | Contains the FUNCTION_MAP with pharmacy operations |
| `config.json` | Configuration for Deepgram, OpenAI, and function definitions |
| `.env` | Contains DEEPGRAM_API_KEY |

#### Running the Server
```bash
source venv/bin/activate
python3 main.py
```

---

## 2. Flask TTS API (tts_app.py)

### Purpose
A REST API that converts text to speech using the Bark model from Hugging Face, with optional Twilio phone call integration.

### Features
- **POST /tts** - Generate speech from text (max 100 characters)
- **GET /health** - Health check endpoint
- **GET /audio/<filename>** - Serve generated audio files
- **GPU Auto-detection** - Automatically uses CUDA if available
- **Twilio Integration** - Can trigger phone calls with generated audio

### How It Works

#### Request Flow
```
Client → Flask API → Bark Model → Audio File → (Optional) Twilio Call → Client
```

1. **Text Validation**: Ensures text is ≤100 characters
2. **Model Generation**: Uses `suno/bark-small` to generate speech
3. **Audio Processing**: Converts to WAV at 24000 Hz sample rate
4. **File Storage**: Saves with UUID filename in `generated_audio/`
5. **Twilio Call** (optional):
   - If `phone_number` provided, creates public URL
   - Initiates Twilio call to play the audio
   - Deletes audio file after call completes
6. **Response**: Returns audio file or success message

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tts` | POST | Generate speech from text |
| `/health` | GET | Health check |
| `/audio/<filename>` | GET | Serve audio file |

#### Example Requests

**Generate Speech Only:**
```bash
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  -o output.wav
```

**Generate Speech AND Trigger Phone Call:**
```bash
curl -X POST https://your-app.onrender.com/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is your pharmacy assistant calling.",
    "phone_number": "+919999999999"
  }'
```

#### Environment Variables (for Twilio)

| Variable | Description |
|----------|-------------|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio phone number |
| `BASE_URL` | Your app's public URL |

---

## Summary

| Project | Technology Stack | Use Case |
|---------|------------------|----------|
| Pharmacy AI Agent | Deepgram, OpenAI GPT-4o-mini, Twilio, WebSockets | Voice-based pharmacy customer service |
| Flask TTS API | Flask, Hugging Face Bark, Twilio | Text-to-speech with phone call capability |

Both applications can work together - the TTS API can be used to generate audio announcements for the pharmacy agent.

