import google.generativeai as genai
import whisper
import os
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image
import PyPDF2
import docx
import mimetypes
import json 

# --- 1. Configuration ---
app = Flask(__name__)
API_KEY = "AIzaSyCxRwMWXgHMH9q0vW-lZj8ML7No_h77Lxk"
# --- 2. Initialize Models ---
gemini_model = None
whisper_model = None
# !! REMOVED !! chat_sessions dictionary is gone

try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini model loaded.")
    else:
        print("Gemini model NOT loaded. API key is missing.")
        print("Please set the GEMINI_API_KEY environment variable.")

    whisper_model = whisper.load_model("base")
    print("Whisper model loaded.")

except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")
    print("The application might not function correctly.")

# --- 3. !! REMOVED !! Chat Session Manager is gone ---

# --- 4. File Processing "Brain" (Unchanged) ---
# This function is exactly the same as before
def process_file_and_query(query, file_storage):
    file_type = file_storage.mimetype
    if file_type.startswith('image/'):
        img = Image.open(file_storage.stream)
        # Return a list of parts for Gemini
        return [query, img]
    elif file_type == 'application/pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(file_storage.stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e: return f"Error: Could not read the PDF file. {e}"
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            doc = docx.Document(file_storage.stream)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e: return f"Error: Could not read the DOCX file. {e}"
    elif file_type.startswith('text/'):
        try:
            text = file_storage.stream.read().decode('utf-8')
        except Exception as e: return f"Error: Could not read the text file. {e}"
    else:
        return f"Error: Unsupported file type ({file_type})."

    # Return a single text string for text-based files
    return f"""You are a helpful assistant. Please answer the user's query based on the following document content.
    --- Document Content ---
    {text}
    ------------------------
    --- User Query ---
    {query}
    """

# --- 5. Define API Routes (All are modified) ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles text-only queries. Is now stateless."""
    if not gemini_model:
        return jsonify({"error": "The Gemini model is not loaded."}), 500
    
    data = request.get_json()
    user_query = data.get('query')
    # !! NEW !! Get history from frontend
    history = data.get('history', []) 
    
    if not user_query:
        return jsonify({"error": "No query provided."}), 400
    
    try:
        # !! MODIFIED !! Start a new chat *every time* with the provided history
        chat_session = gemini_model.start_chat(history=history)
        response = chat_session.send_message(user_query)
        # !! MODIFIED !! No longer returns chat_id
        return jsonify({"reply": response.text})
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/ask_with_file', methods=['POST'])
def ask_with_file():
    """Handles queries that come with a file. Is now stateless."""
    if not gemini_model:
        return jsonify({"error": "The Gemini model is not loaded."}), 500
    if 'file' not in request.files or 'query' not in request.form:
        return jsonify({"error": "Missing file or query."}), 400

    query = request.form.get('query')
    # !! NEW !! Get history from form data
    history_str = request.form.get('history', '[]')
    history = json.loads(history_str)
    file = request.files.get('file')

    try:
        chat_session = gemini_model.start_chat(history=history) # !! MODIFIED !!
        prompt_or_list = process_file_and_query(query, file)
        
        if isinstance(prompt_or_list, str) and prompt_or_list.startswith("Error:"):
            return jsonify({"error": prompt_or_list}), 400
            
        response = chat_session.send_message(prompt_or_list) # !! MODIFIED !!
        return jsonify({"reply": response.text}) # !! MODIFIED !!
    except Exception as e:
        print(f"Error during multimodal API call: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/transcribe_and_ask', methods=['POST'])
def transcribe_and_ask():
    """Handles audio queries, with or without a file. Is now stateless."""
    if not whisper_model or not gemini_model:
        return jsonify({"error": "A model is not loaded."}), 500
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found."}), 400

    audio_file = request.files['audio']
    file_storage = request.files.get('file')
    # !! NEW !! Get history from form data
    history_str = request.form.get('history', '[]')
    history = json.loads(history_str)
    
    temp_filename = "temp_audio.wav"
    audio_file.save(temp_filename)

    try:
        transcription_result = whisper_model.transcribe(temp_filename, fp16=False)
        transcribed_text = transcription_result["text"]

        if not transcribed_text.strip():
            return jsonify({
                "transcription": "(Silence or unrecognized audio)",
                "reply": "I'm sorry, I didn't catch that. Could you please repeat?"
            })

        # !! MODIFIED !!
        chat_session = gemini_model.start_chat(history=history)
        
        if file_storage:
            prompt_or_list = process_file_and_query(transcribed_text, file_storage)
            if isinstance(prompt_or_list, str) and prompt_or_list.startswith("Error:"):
                return jsonify({"transcription": transcribed_text, "error": prompt_or_list}), 400
            response = chat_session.send_message(prompt_or_list)
        else:
            response = chat_session.send_message(transcribed_text)
        
        return jsonify({
            "transcription": transcribed_text,
            "reply": response.text
            # !! MODIFIED !! No longer returns chat_id
        })
    except Exception as e:
        print(f"Error during transcription or API call: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- 6. Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)