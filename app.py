from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_ml_model, predict_phishing

app = Flask(__name__)
CORS(app)

model, tokenizer = load_ml_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data.get('url')
        print("------------------------------------->" + url)

        if url is not None:
            prediction = predict_phishing(url, model, tokenizer)
            print("--------------------------------->" + str(prediction))  

            result = "ham" if prediction < 0.5 else "spam"
            print(result)
            return jsonify({'result': result}), 200 
        else:
            return jsonify({'error': "Invalid request. 'url' parameter is missing."}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
