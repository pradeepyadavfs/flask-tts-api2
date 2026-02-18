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
- Logging
- Rate limiting
- Input sanitization
- Metrics tracking
"""

import os
import re
import uuid
import logging
import torch
import soundfile as sf
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from transformers import AutoProcessor, BarkModel
from twilio.rest import Client
from twilio.twiml import VoiceResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Rate limiting: 5 requests per IP per minute
rate_limit_store = defaultdict(list)
RATE_LIMIT = 5
RATE_LIMIT_WINDOW = 60  # seconds

# Metrics counters
total_calls_made = 0
total_tts_generated = 0


def load_model():
    """Load the Bark model and processor at startup with GPU support."""
    global processor, model, device, model_loaded
    
    logger.info("Loading Bark model and processor...")
    
    # Detect GPU/CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU detected! Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected. Using CPU for inference.")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small")
    
    # Move model to the detected device once at startup for performance
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    model_loaded = True
    logger.info("Model loaded successfully!")


def create_audio_folder():
    """Create the audio output folder if it doesn't exist."""
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
        logger.info(f"Created audio folder: {AUDIO_FOLDER}")


def sanitize_text(text):
    """
    Sanitize input text by stripping whitespace and validating content.
    
    Args:
        text: Raw input text
        
    Returns:
        tuple: (is_valid, sanitized_text, error_message)
    """
    # Strip extra whitespace
    sanitized = text.strip()
    
    # Check if empty after strip
    if not sanitized:
        return False, "", "Text cannot be empty"
    
    # Check if only special characters (allow letters, numbers, spaces, basic punctuation)
    # This regex matches strings that have at least one alphanumeric character
    if not re.search(r'[a-zA-Z0-9]', sanitized):
        return False, "", "Text must contain at least one alphanumeric character"
    
    return True, sanitized, None


def check_rate_limit(ip):
    """
    Check if IP has exceeded rate limit.
    
    Args:
        ip: Client IP address
        
    Returns:
        tuple: (is_allowed, error_response)
    """
    now = datetime.now()
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    # Clean old entries
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if t > cutoff]
    
    # Check limit
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return False, jsonify({
            "error": "Rate limit exceeded",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": f"Maximum {RATE_LIMIT} requests per minute allowed",
            "retry_after": RATE_LIMIT_WINDOW
        }), 429
    
    # Add current request
    rate_limit_store[ip].append(now)
    return True, None, None


def create_error_response(error_code, message, status_code=400):
    """
    Create a structured JSON error response.
    
    Args:
        error_code: Machine-readable error code
        message: Human-readable error message
        status_code: HTTP status code
        
    Returns:
        tuple: (JSON response, status code)
    """
    return jsonify({
        "error": error_code,
        "message": message
    }), status_code


def trigger_twilio_call(phone_number, audio_url):
    """
    Trigger a Twilio phone call to play the generated audio.
    
    Args:
        phone_number: The phone number to call
        audio_url: Public URL of the generated audio file
    
    Returns:
        dict: Success message with call SID or error
    """
    global total_calls_made
    
    try:
        # Get Twilio credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_phone]):
            logger.error("Twilio credentials not configured")
            return {"error": "Twilio credentials not configured", "error_code": "TWILIO_NOT_CONFIGURED"}, 500
        
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
        
        total_calls_made += 1
        logger.info(f"Call initiated: SID={call.sid}, to={phone_number}")
        
        return {
            "success": True,
            "call_sid": call.sid,
            "message": f"Call initiated to {phone_number}"
        }, 200
        
    except Exception as e:
        logger.error(f"Twilio call failed: {str(e)}")
        return {"error": f"Failed to trigger call: {str(e)}", "error_code": "TWILIO_ERROR"}, 500


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
                logger.info(f"Delayed cleanup: deleted {file_path}")
        except Exception as e:
            logger.error(f"Error in delayed cleanup: {e}")
    
    # Run in background thread to avoid blocking main thread
    thread = threading.Thread(target=_deleter, daemon=True)
    thread.start()


def schedule_file_deletion(filepath):
    """Schedule file deletion after response is sent using delayed cleanup."""
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
    global total_tts_generated
    
    client_ip = request.remote_addr
    logger.info(f"TTS request received from {client_ip}")
    
    # Check rate limit
    is_allowed, error_response, status_code = check_rate_limit(client_ip)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return error_response, status_code
    
    # Check if request has JSON content
    if not request.is_json:
        logger.warning(f"Invalid content type from {client_ip}")
        return create_error_response("INVALID_CONTENT_TYPE", "Content-Type must be application/json", 400)
    
    # Get JSON data
    data = request.get_json()
    
    # Validate text field exists
    if 'text' not in data:
        return create_error_response("MISSING_FIELD", "Missing required field: 'text'", 400)
    
    text = data.get('text', '')
    phone_number = data.get('phone_number', None)  # Optional phone number
    
    # Sanitize input text
    is_valid, sanitized_text, error_message = sanitize_text(text)
    if not is_valid:
        logger.warning(f"Invalid text input from {client_ip}: {error_message}")
        return create_error_response("INVALID_TEXT", error_message, 400)
    
    # Validate text length (max 100 characters)
    if len(sanitized_text) > MAX_TEXT_LENGTH:
        return create_error_response(
            "TEXT_TOO_LONG",
            f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters (current: {len(sanitized_text)})",
            400
        )
    
    logger.info(f"Processing TTS request: text_length={len(sanitized_text)}, phone={phone_number is not None}")
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    
    try:
        # Process input text
        inputs = processor(sanitized_text, voice_preset="v2_en_speaker_6")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate speech using torch.no_grad() for inference
        with torch.no_grad():
            output = model.generate(**inputs)
        
        # Convert tensor to numpy array
        audio_array = output.cpu().numpy().squeeze()
        
        # Save as WAV file with 24000 sample rate
        sf.write(filepath, audio_array, SAMPLE_RATE)
        
        total_tts_generated += 1
        logger.info(f"TTS generated successfully: {filename}")
        
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
                "text": sanitized_text,
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
        error_str = str(e)
        
        # Determine error type for logging
        if "model" in error_str.lower() or "processor" in error_str.lower():
            logger.error(f"Model error: {error_str}")
            error_code = "MODEL_ERROR"
        else:
            logger.error(f"TTS generation failed: {error_str}")
            error_code = "GENERATION_ERROR"
        
        # Clean up file if it was created but generation failed
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return create_error_response(error_code, f"Failed to generate speech: {error_str}", 500)


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
        return create_error_response("FILE_NOT_FOUND", f"Audio file '{filename}' not found", 404)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service is running and model status."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Metrics endpoint to track API usage.
    
    Returns:
        JSON with total_calls_made and total_tts_generated
    """
    return jsonify({
        "total_calls_made": total_calls_made,
        "total_tts_generated": total_tts_generated
    })


if __name__ == '__main__':
    # Create audio folder
    create_audio_folder()
    
    # Load model at startup
    load_model()
    
    # Note: For production, use Gunicorn instead of Flask dev server
    # Run with: gunicorn -w 1 -b 0.0.0.0:$PORT tts_app:app
    # The app object is exposed for Gunicorn import

