"""
Simple TTS + Twilio Call API
Run locally: python tts_call.py
"""

import os
import uuid
from flask import Flask, request, jsonify
from transformers import AutoProcessor, BarkModel
import soundfile as sf
import torch

# ============== CONFIGURATION ==============
# Replace these with your Twilio credentials
TWILIO_ACCOUNT_SID = "your_account_sid_here"
TWILIO_AUTH_TOKEN = "your_auth_token_here"
TWILIO_PHONE_NUMBER = "+1234567890"  # Your Twilio phone number
# ===========================================

app = Flask(__name__)

# Configuration
SAMPLE_RATE = 24000
AUDIO_FOLDER = "generated_audio"
MAX_TEXT_LENGTH = 100

# Load model at startup
print("Loading TTS model...")
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded! Using: {device}")

# Create audio folder
os.makedirs(AUDIO_FOLDER, exist_ok=True)


def generate_tts(text):
    """Generate speech from text using Bark model."""
    inputs = processor(text, voice_preset="v2_en_speaker_6")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(**inputs)
    
    audio_array = output.cpu().numpy().squeeze()
    return audio_array


def make_call(to_number, audio_path):
    """Make a Twilio call and play the audio file."""
    from twilio.rest import Client
    from twilio.twiml import VoiceResponse
    
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    twiml = VoiceResponse()
    twiml.play(audio_path)
    
    call = client.calls.create(
        to=to_number,
        from_=TWILIO_PHONE_NUMBER,
        twiml=str(twiml)
    )
    
    return call.sid


@app.route('/tts-call', methods=['POST'])
def tts_call():
    """Generate TTS and initiate a phone call."""
    
    # Check JSON content
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    # Validate required fields
    if 'text' not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400
    
    if 'to' not in data:
        return jsonify({"error": "Missing required field: 'to'"}), 400
    
    text = data.get('text', '').strip()
    to_number = data.get('to')
    
    # Validate text length
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Text exceeds {MAX_TEXT_LENGTH} characters (got {len(text)})"
        }), 400
    
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    # Generate audio
    print(f"Generating TTS for: {text}")
    audio_array = generate_tts(text)
    
    # Save to file
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    sf.write(filepath, audio_array, SAMPLE_RATE)
    print(f"Audio saved: {filepath}")
    
    try:
        # Make the call - use file path for local playback
        call_sid = make_call(to_number, filepath)
        print(f"Call initiated: {call_sid}")
        
        return jsonify({
            "success": True,
            "message": "Call initiated",
            "call_sid": call_sid,
            "text": text,
            "to": to_number
        })
        
    except Exception as e:
        return jsonify({"error": f"Call failed: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

