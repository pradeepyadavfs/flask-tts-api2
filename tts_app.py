
# Requirements:
# flask
# torch
# transformers
# scipy
# soundfile
# gunicorn
# twilio

"""
Text-to-Speech API using Hugging Face Bark Model
Flask application that converts text to speech using suno/bark-small model
AND triggers Twilio phone calls with the generated audio

Production-ready implementation with:
- GPU auto-detection
- Unique filename generation with auto-cleanup
- Proper error handling
- Optimized performance
- Twilio phone call integration
"""

import os
import uuid
import torch
import soundfile as sf
import threading
import time
from flask import Flask, request, jsonify, send_file
from transformers import AutoProcessor, BarkModel
from twilio.rest import Client
from twilio.twiml import VoiceResponse

# Initialize Flask app (production-ready, debug=False)
app = Flask(__name__)

# Production settings: Limit max request size to 1MB to avoid abuse
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# Global variables for model and processor (loaded once at startup)
processor = None
model = None
device = None
model_loaded = False

# Configuration
SAMPLE_RATE = 24000
AUDIO_FOLDER = "generated_audio"
MAX_TEXT_LENGTH = 100


def load_model():
    """Load the Bark model and processor at startup with GPU support."""
    global processor, model, device, model_loaded
    
    print("Loading Bark model and processor...")
    
    # Detect GPU/CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected! Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU for inference.")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")
    
    # Move model to the detected device once at startup for performance
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    model_loaded = True
    print("Model loaded successfully!")


def create_audio_folder():
    """Create the audio output folder if it doesn't exist."""
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
        print(f"Created audio folder: {AUDIO_FOLDER}")


def trigger_twilio_call(phone_number, audio_url):
    """
    Trigger a Twilio phone call to play the generated audio.
    
    Args:
        phone_number: The phone number to call
        audio_url: Public URL of the generated audio file
    
    Returns:
        dict: Success message with call SID or error
    """
    try:
        # Get Twilio credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_phone]):
            return {"error": "Twilio credentials not configured"}, 500
        
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Create TwiML response with the audio URL
        twiml = VoiceResponse()
        twiml.play(audio_url)
        
        # Make the call
        call = client.calls.create(
            to=phone_number,
            from_=twilio_phone,
            twiml=str(twiml)
        )
        
        return {
            "success": True,
            "call_sid": call.sid,
            "message": f"Call initiated to {phone_number}"
        }, 200
        
    except Exception as e:
        return {"error": f"Failed to trigger call: {str(e)}"}, 500


def delete_file_later(file_path, delay=90):
    """
    Delete a file after a specified delay in a background thread.
    Gives Twilio enough time to fetch the audio before cleanup.
    
    Args:
        file_path: Path to the file to delete
        delay: Seconds to wait before deletion (default 90)
    """
    def _deleter():
        try:
            time.sleep(delay)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Delayed cleanup: deleted {file_path}")
        except Exception as e:
            print(f"Error in delayed cleanup: {e}")
    
    # Run in background thread to avoid blocking main thread
    thread = threading.Thread(target=_deleter, daemon=True)
    thread.start()


def schedule_file_deletion(filepath):
    """Schedule file deletion after response is sent (legacy - using new delayed cleanup)."""
    delete_file_later(filepath, delay=90)


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    POST endpoint for Text-to-Speech conversion.
    
    Expected JSON body: 
    {
        "text": "your input text here",
        "phone_number": "+91XXXXXXXXXX"  (optional - triggers call)
    }
    
    Returns: 
    - If phone_number provided: JSON response with call status
    - Otherwise: audio/wav file
    """
    
    # Check if request has JSON content
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400
    
    # Get JSON data
    data = request.get_json()
    
    # Validate text field exists
    if 'text' not in data:
        return jsonify({
            "error": "Missing required field: 'text'"
        }), 400
    
    text = data.get('text', '')
    phone_number = data.get('phone_number', None)  # Optional phone number
    
    # Validate text is not empty
    if not text or not text.strip():
        return jsonify({
            "error": "Text field cannot be empty"
        }), 400
    
    # Validate text length (max 100 characters)
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters (current: {len(text)})"
        }), 400
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    
    try:
        # Process input text
        inputs = processor(text, voice_preset="v2_en_speaker_6")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate speech using torch.no_grad() for inference
        with torch.no_grad():
            output = model.generate(**inputs)
        
        # Convert tensor to numpy array
        audio_array = output.cpu().numpy().squeeze()
        
        # Save as WAV file with 24000 sample rate
        sf.write(filepath, audio_array, SAMPLE_RATE)
        
        # If phone_number is provided, trigger Twilio call
        if phone_number:
            # Dynamically detect base URL from request
            base_url = request.host_url.rstrip("/")
            
            # Create public URL for the audio file
            audio_url = f"{base_url}/audio/{filename}"
            
            # Trigger the Twilio call
            call_result, status_code = trigger_twilio_call(phone_number, audio_url)
            
            # Schedule delayed cleanup (90 seconds) to give Twilio time to fetch audio
            delete_file_later(filepath, delay=90)
            
            if status_code != 200:
                return jsonify(call_result), status_code
            
            return jsonify({
                "success": True,
                "message": "Audio generated and call initiated",
                "text": text,
                "call_sid": call_result.get("call_sid"),
                "phone_number": phone_number
            }), 200
        
        # If no phone number, return the audio file directly
        # Set up automatic file deletion after response is sent
        schedule_file_deletion(filepath)
        
        # Return the generated audio file
        return send_file(
            filepath,
            mimetype="audio/wav",
            as_attachment=False
        )
        
    except Exception as e:
        # Clean up file if it was created but generation failed
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            "error": f"Failed to generate speech: {str(e)}"
        }), 500


@app.route('/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """
    GET endpoint to serve generated audio files.
    
    Args:
        filename: Name of the audio file to serve
    
    Returns:
        audio/wav file or 404 if not found
    """
    # Prevent directory traversal attacks
    filename = os.path.basename(filename)
    filepath = os.path.join(AUDIO_FOLDER, filename)
    
    if os.path.exists(filepath):
        # Schedule file deletion after serving
        schedule_file_deletion(filepath)
        
        return send_file(
            filepath,
            mimetype="audio/wav",
            as_attachment=False
        )
    else:
        return jsonify({
            "error": f"Audio file '{filename}' not found"
        }), 404


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service is running and model status."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })


if __name__ == '__main__':
    # Create audio folder
    create_audio_folder()
    
    # Load model at startup
    load_model()
    
    # Note: For production, use Gunicorn instead of Flask dev server
    # Run with: gunicorn -w 1 -b 0.0.0.0:$PORT tts_app:app
    # The app object is exposed for Gunicorn import

