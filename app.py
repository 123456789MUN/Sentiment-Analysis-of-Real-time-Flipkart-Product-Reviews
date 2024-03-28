from flask import Flask, render_template, request

app = Flask(__name__)

# Dummy list to store submitted reviews
reviews = []

@app.route('/')
def index():
    return render_template('review_form.html')

@app.route('/submit_review', methods=['POST'])
def submit_review():
    name = request.form['name']
    rating = request.form['rating']
    review_text = request.form['review']
    
    # Convert rating to integer
    rating = int(rating)

    # Append the new review to the reviews list
    reviews.append({'name': name, 'rating': rating, 'review': review_text})

    return render_template('result.html', reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)
