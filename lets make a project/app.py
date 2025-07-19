from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import random

app = Flask(__name__)

# Global variables for the model and feature processing
model = None
feature_columns = None
training_data = None

def load_model():
    """Load the trained model and feature columns"""
    global model, feature_columns, training_data
    
    if os.path.exists('titanic_model.pkl'):
        with open('titanic_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_columns = model_data['feature_columns']
        
        # Load training data for random passenger generation
        if os.path.exists('data/train.csv'):
            training_data = pd.read_csv('data/train.csv')
        
        return True
    return False

def get_random_passenger():
    """Get a random passenger from the training data"""
    if training_data is None:
        return None
    
    # Get a random row from training data
    random_row = training_data.sample(n=1).iloc[0]
    
    # Extract title from name
    name_parts = random_row['Name'].split(', ')
    if len(name_parts) >= 2:
        title_part = name_parts[1].split('.')[0]
        title = title_part  # Don't add period, just use the title part
    else:
        title = 'Mr'
    
    # Clean up title (remove extra spaces and ensure proper format)
    title = title.strip()
    
    # Create passenger data
    passenger_data = {
        'passenger_id': int(random_row['PassengerId']),
        'name': random_row['Name'],
        'title': title,
        'sex': random_row['Sex'],
        'age': random_row['Age'] if pd.notna(random_row['Age']) else None,
        'pclass': int(random_row['Pclass']),
        'sibsp': int(random_row['SibSp']),
        'parch': int(random_row['Parch']),
        'ticket': random_row['Ticket'],
        'fare': random_row['Fare'] if pd.notna(random_row['Fare']) else None,
        'embarked': random_row['Embarked'] if pd.notna(random_row['Embarked']) else 'S',
        'actual_survived': bool(random_row['Survived'])
    }
    
    return passenger_data

def process_passenger_data(passenger_data):
    """Process passenger data to match the model's expected format"""
    # Create a DataFrame with the passenger data
    df = pd.DataFrame([passenger_data])
    
    # Title mapping from the original notebook
    Title_Dictionary = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Jonkheer": "Royalty", "Don": "Royalty", "Sir": "Royalty",
        "Dr": "Officer", "Rev": "Officer", "the Countess": "Royalty",
        "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr",
        "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Royalty"
    }
    
    # Process title - use the provided title directly
    # Handle both 'Title' and 'title' column names
    title_col = 'Title' if 'Title' in df.columns else 'title'
    df['Title'] = df[title_col].replace(Title_Dictionary).fillna('Mr')
    
    # Handle column name variations (capital vs lowercase)
    age_col = 'Age' if 'Age' in df.columns else 'age'
    fare_col = 'Fare' if 'Fare' in df.columns else 'fare'
    embarked_col = 'Embarked' if 'Embarked' in df.columns else 'embarked'
    sex_col = 'Sex' if 'Sex' in df.columns else 'sex'
    sibsp_col = 'SibSp' if 'SibSp' in df.columns else 'sibsp'
    parch_col = 'Parch' if 'Parch' in df.columns else 'parch'
    pclass_col = 'Pclass' if 'Pclass' in df.columns else 'pclass'
    ticket_col = 'Ticket' if 'Ticket' in df.columns else 'ticket'
    
    # Process age (use median if missing)
    if pd.isna(df[age_col].iloc[0]):
        df['Age'] = 28.0  # Median age from training data
    else:
        df['Age'] = df[age_col]
    
    # Process fare (use median if missing)
    if pd.isna(df[fare_col].iloc[0]):
        df['Fare'] = 14.4542  # Median fare from training data
    else:
        df['Fare'] = df[fare_col]
    
    # Process embarked (use 'S' if missing)
    if pd.isna(df[embarked_col].iloc[0]):
        df['Embarked'] = 'S'
    else:
        df['Embarked'] = df[embarked_col]
    
    # Process sex
    df['Sex'] = df[sex_col].replace({'male': 1, 'female': 0})
    
    # Create family size
    df['FamilySize'] = df[sibsp_col] + df[parch_col] + 1
    df['Singleton'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)
    
    # Create dummy variables for categorical features
    # Title dummies
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)
    
    # Embarked dummies
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    

    
    # Pclass dummies
    df['Pclass'] = df[pclass_col]
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, pclass_dummies], axis=1)
    
    # Process ticket
    def clean_ticket(ticket):
        ticket = str(ticket).replace('.', '').replace('/', '')
        ticket = ticket.split()
        ticket = [t.strip() for t in ticket]
        ticket = [t for t in ticket if not t.isdigit()]
        return ticket[0] if len(ticket) > 0 else 'XXX'
    
    df['Ticket'] = df[ticket_col].apply(clean_ticket)
    ticket_dummies = pd.get_dummies(df['Ticket'], prefix='Ticket')
    df = pd.concat([df, ticket_dummies], axis=1)
    
    # Select only the features that the model expects
    if feature_columns is not None:
        # Ensure all expected columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the expected columns in the right order
        df = df[feature_columns]
    
    return df

@app.route('/')
def home():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = {
            'Pclass': int(request.form['pclass']),
            'Name': request.form['name'],
            'Title': request.form['title'],
            'Sex': request.form['sex'],
            'Age': float(request.form['age']) if request.form['age'] else None,
            'SibSp': int(request.form['sibsp']),
            'Parch': int(request.form['parch']),
            'Ticket': request.form['ticket'],
            'Fare': float(request.form['fare']) if request.form['fare'] else None,
            'Embarked': request.form['embarked']
        }
        
        # Process the data
        processed_data = process_passenger_data(data)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            result = {
                'survived': bool(prediction),
                'survival_probability': float(probability[1]),
                'death_probability': float(probability[0])
            }
        else:
            result = {
                'error': 'Model not loaded. Please train the model first.'
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    """About page explaining the project"""
    return render_template('about.html')

@app.route('/random')
def random_passenger():
    """Generate and display a random passenger for testing"""
    passenger = get_random_passenger()
    if passenger:
        return render_template('random_passenger.html', passenger=passenger)
    else:
        return "No training data available for random passenger generation."

@app.route('/predict_random', methods=['POST'])
def predict_random():
    """Handle prediction for random passenger"""
    try:
        # Get passenger data from request
        passenger_data = request.get_json()
        
        # Process the data for prediction
        processed_data = process_passenger_data(passenger_data)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            # Get user's prediction
            user_prediction = passenger_data.get('user_prediction', None)
            
            # Compare user's prediction with actual result
            user_correct = user_prediction == passenger_data['actual_survived'] if user_prediction is not None else False
            
            # Debug: Print the values to understand what's happening
            print(f"DEBUG - Passenger: {passenger_data.get('name', 'Unknown')}")
            print(f"DEBUG - Model prediction: {bool(prediction)} (survival prob: {probability[1]:.3f})")
            print(f"DEBUG - Actual survived: {passenger_data['actual_survived']}")
            print(f"DEBUG - User prediction: {user_prediction}")
            print(f"DEBUG - User correct: {user_correct}")
            
            result = {
                'predicted_survived': bool(prediction),
                'survival_probability': float(probability[1]),
                'death_probability': float(probability[0]),
                'actual_survived': passenger_data['actual_survived'],
                'user_prediction': user_prediction,
                'correct': user_correct
            }
        else:
            result = {
                'error': 'Model not loaded. Please train the model first.'
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if load_model():
        print("Model loaded successfully!")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        print("Failed to load model. Please train the model first.") 