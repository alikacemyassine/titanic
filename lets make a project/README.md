# Titanic Survival Predictor

A web application that predicts whether a passenger would have survived the Titanic disaster using machine learning.

## Features

- **Interactive Web Interface**: Beautiful, modern UI for entering passenger details
- **Machine Learning Model**: Random Forest classifier with 81.34% accuracy
- **Real-time Predictions**: Instant survival probability calculations
- **Feature Engineering**: Advanced data processing based on the original analysis

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**:
   ```bash
   python train_and_save_model.py
   ```

3. **Run the web application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## How to Use

1. Fill in the passenger details:
   - **Full Name**: Enter the passenger's full name (e.g., "Smith, Mr. John")
   - **Gender**: Select male or female
   - **Age**: Enter the passenger's age
   - **Passenger Class**: Choose 1st, 2nd, or 3rd class
   - **Ticket Fare**: Enter the ticket price
   - **Family Members**: Number of siblings/spouses and parents/children
   - **Port of Embarkation**: Select Southampton, Cherbourg, or Queenstown
   - **Ticket Number**: Enter the ticket number
   - **Cabin**: Enter the cabin number (optional)

2. Click "Predict Survival" to get the result

3. View the prediction with survival/death probabilities

## Model Details

The model uses a Random Forest classifier trained on the famous Titanic dataset with the following features:

- **Demographics**: Age, gender, passenger class
- **Family Information**: Siblings, spouses, parents, children
- **Travel Details**: Ticket fare, cabin, embarkation port
- **Derived Features**: Title extraction, family size calculations

## Technical Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Model Storage**: Pickle

## Files Structure

```
├── app.py                          # Flask web application
├── train_and_save_model.py         # Model training script
├── requirements.txt                # Python dependencies
├── titanic_model.pkl              # Trained model (generated)
├── templates/
│   ├── index.html                 # Main prediction page
│   └── about.html                 # About page
├── data/
│   ├── train.csv                  # Training dataset
│   └── test.csv                   # Test dataset
└── README.md                      # This file
```

## Model Performance

- **Accuracy**: 81.34% (top 4% on Kaggle leaderboard)
- **Algorithm**: Random Forest Classifier
- **Feature Selection**: Automatic feature importance selection
- **Cross-validation**: 5-fold stratified cross-validation

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

