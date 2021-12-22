from types import MethodDescriptorType
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.models import load_model
import pickle

nltk.download('stopwords')

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    #Lower
    text = request.form['text'].lower()

    #Tokenizing 
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tokens = token.tokenize(text)
    
    #Number removal
    num_remove = [i for i in tokens if i.isalpha()]

    #punctuation and stopwords removal
    st = stopwords.words('english')
    punctuations = string.punctuation
    st_pc_remove = [word for word in num_remove if word.lower() not in st and word not in punctuations]

    #Stemming
    stemmer = PorterStemmer()
    truncated = []
    for i in st_pc_remove:
        stemmed = stemmer.stem(i)
        truncated.append(stemmed)
    
    sentence = ' '.join(truncated)


    #Padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequence = tokenizer.texts_to_sequences(sentence)
    padding = pad_sequences(sequence, maxlen = 23)

    #Reshape vector 
    X = padding.reshape(1,-1)
    
    #Prediction 
    prediction = model.predict(X)

    return render_template('index.html', prediction_text = 'Probability of Negative {}, Probability of Positive {}'.format(prediction[0][0], prediction[0][1]))

# @app.route('/results' , methods=['POST'])
# def results():
#     data = request.get_json(force=True) #Retrieve data as json
#     prediction = model.predict([])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug = True, port=8002)