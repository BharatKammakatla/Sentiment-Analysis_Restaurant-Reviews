from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pickle

# Load the Multinomial Navie Bayes and TFID vectorizer object from local
classifier = pickle.load(open('restaurant-sentiment-mnb-model.pkl', 'rb'))
tf = pickle.load(open('tf-transform.pkl', 'rb'))


app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        tf_vect = tf.transform(data).toarray()
        prediction = classifier.predict(tf_vect)
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    #bootstrap.init_app(app)  # Initialize bootstrap
    app.run(debug=True)
