from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = vectorizer.transform([message])
        prediction = model.predict(cleaned_message)[0]
        prediction = 'Spam' if prediction == 1 else 'Legitimate'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
