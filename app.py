from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the data
df = pd.read_csv('data_GEN6.csv')

# Prepare the model
y = df['FATAL']
X = df.drop(['FATAL', 'INJ_SEV', 'INJ_LEVEL'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=476, max_depth=50, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return "Fatality Predictor API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
