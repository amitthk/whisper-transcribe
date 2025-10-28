from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel
import os
from pathlib import Path
import uuid

app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, cors_allowed_origins="*")

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model once
model = WhisperModel("medium", device="cuda")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.mp3'):
        return jsonify({'error': 'Only MP3 files are supported'}), 400
    
    # Save uploaded file with unique name
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.mp3"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Get optional output filename from form data
    output_filename = request.form.get('output_filename', '')
    
    # Start transcription in background
    socketio.start_background_task(transcribe_file, filepath, file_id, output_filename)
    
    return jsonify({'file_id': file_id, 'message': 'Upload successful, transcription started'})

def transcribe_file(filepath, file_id, output_filename):
    try:
        segments, info = model.transcribe(filepath)
        full_transcript = []
        
        for segment in segments:
            text = segment.text.strip()
            full_transcript.append(text)
            # Emit each segment as it's processed
            socketio.emit('transcription_segment', {
                'file_id': file_id,
                'text': text,
                'start': segment.start,
                'end': segment.end
            })
        
        # Save to file if output filename provided
        if output_filename:
            output_path = Path(app.config['UPLOAD_FOLDER']) / f"{output_filename}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_transcript))
            
            socketio.emit('transcription_complete', {
                'file_id': file_id,
                'saved_file': str(output_path),
                'full_transcript': '\n'.join(full_transcript)
            })
        else:
            socketio.emit('transcription_complete', {
                'file_id': file_id,
                'full_transcript': '\n'.join(full_transcript)
            })
        
    except Exception as e:
        socketio.emit('transcription_error', {
            'file_id': file_id,
            'error': str(e)
        })
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8282)
