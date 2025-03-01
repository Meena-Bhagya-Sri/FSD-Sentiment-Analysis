import os
import pickle
import re
import matplotlib
matplotlib.use('Agg')  # Fix for Matplotlib GUI error
import matplotlib.pyplot as plt

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from preprocess import preprocess_text

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Review model
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)

# Load the trained model and vectorizer
model = pickle.load(open('nb_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)

    sentiment = "good😊" if prediction[0] == 1 else "bad🧐"

    # Save to database
    new_review = Review(text=review, sentiment=sentiment)
    db.session.add(new_review)
    db.session.commit()

    return render_template('index.html', prediction=sentiment)

@app.route('/dashboard')
def dashboard():
    reviews = Review.query.all()
    good_count = Review.query.filter_by(sentiment="good😊").count()
    bad_count = Review.query.filter_by(sentiment="bad🧐").count()

    # Ensure 'static' directory exists
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Generate pie chart
    labels = ['Good', 'Bad']
    values = [good_count, bad_count]
    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title("Sentiment Distribution")

    # Save chart
    chart_path = os.path.join(static_dir, 'chart.png')
    plt.savefig(chart_path)
    plt.close()

    return render_template('dashboard.html', chart='static/chart.png', reviews=reviews)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
