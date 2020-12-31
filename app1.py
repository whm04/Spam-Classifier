from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('hello.html')

@app.route('/predict',methods=['POST'])
def predict_fun():
	
    NB_spam_model = open('NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    
    cv_model = open('cv.pkl', 'rb')
    cv = joblib.load(cv_model)
    
    if request.method == 'POST':
        message = request.form['text']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = ["not spam" ,"Spam"][clf.predict(vect)[0]]
    return render_template('hello.html',positive = my_prediction,pred=message)


if __name__ == '__main__':
	app.run()
