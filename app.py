from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_mfcc, load_scaler, load_label_encoder

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle et les utilitaires
model = load_model('saved_models/emotion_model.keras')
scaler = load_scaler('saved_models/scaler.npy')
le = load_label_encoder('saved_models/label_encoder.npy')

# Dictionnaire des émotions (adapté à votre modèle)
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Vérifier si le fichier est présent
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Traitement du fichier audio
            try:
                # Extraire les features
                mfcc = extract_mfcc(filepath)
                
                # Normalisation
                mfcc = scaler.transform([mfcc])
                
                # Prédiction
                pred = model.predict(mfcc)
                pred_class = np.argmax(pred, axis=1)
                
                # Obtenir l'émotion
                emotion_code = le.inverse_transform(pred_class)[0]
                emotion = EMOTION_MAP.get(emotion_code, 'unknown')
                
                # Probabilités
                emotion_probs = {
                    EMOTION_MAP.get(le.classes_[i], 'unknown'): float(pred[0][i]) 
                    for i in range(len(le.classes_))
                }
                
                return jsonify({
                    'emotion': emotion,
                    'probabilities': emotion_probs,
                    'audio_file': filename
                })
                
            except Exception as e:
                return jsonify({'error': str(e)})
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)