import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

def load_ml_model():
    model = load_model('phishing_model.h5')

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

def predict_phishing(url, model, tokenizer):
 
    sequences = tokenizer.texts_to_sequences([url])
    input_sequences = sequence.pad_sequences(sequences, maxlen=150)

    prediction = model.predict(input_sequences)[0][0]

    return prediction
