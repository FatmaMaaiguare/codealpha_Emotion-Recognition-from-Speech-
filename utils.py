import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(filename, duration=3, offset=0.5, n_mfcc=40):
    """Extrait les coefficients MFCC d'un fichier audio"""
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

def save_scaler(scaler, filepath):
    """Sauvegarde le scaler"""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(filepath):
    """Charge le scaler"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_label_encoder(le, filepath):
    """Sauvegarde le label encoder"""
    np.save(filepath, le.classes_)

def load_label_encoder(filepath):
    """Charge le label encoder"""
    le = LabelEncoder()
    le.classes_ = np.load(filepath, allow_pickle=True)
    return le