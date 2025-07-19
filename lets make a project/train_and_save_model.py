"""
Script to train and save the Titanic survival prediction model
Based on the analysis from article_1.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
import warnings
warnings.filterwarnings('ignore')

def get_combined_data():
    """Load and combine train and test data"""
    # reading train data
    train = pd.read_csv('./data/train.csv')
    
    # reading test data
    test = pd.read_csv('./data/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop(['Survived'], axis=1, inplace=True)
    
    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = pd.concat([train, test], ignore_index=True)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], axis=1, inplace=True)
    
    return combined, targets

def get_titles(combined):
    """Extract and process titles from names"""
    Title_Dictionary = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Jonkheer": "Royalty", "Don": "Royalty", "Sir": "Royalty",
        "Dr": "Officer", "Rev": "Officer", "the Countess": "Royalty",
        "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr": "Mr",
        "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Royalty"
    }
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    print('Processing Title : ok')
    return combined

def process_age(combined):
    """Process age with imputation based on groups"""
    # Calculate median ages by group for imputation
    grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train['Age'].median().reset_index()
    
    def fill_age(row):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) & 
            (grouped_median_train['Title'] == row['Title']) & 
            (grouped_median_train['Pclass'] == row['Pclass'])
        ) 
        return grouped_median_train[condition]['Age'].values[0]

    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    print('Processing age : ok')
    return combined

def process_names(combined):
    """Process names and create title dummy variables"""
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    
    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)
    
    print('Processing names : ok')
    return combined

def process_fares(combined):
    """Process fare values"""
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    print('Processing fare : ok')
    return combined

def process_embarked(combined):
    """Process embarked values"""
    # two missing embarked values - filling them with the most frequent one in the train set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    print('Processing embarked : ok')
    return combined

def process_cabin(combined):
    """Process cabin values"""
    # replacing missing cabins with U (for Unknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    print('Processing cabin : ok')
    return combined

def process_sex(combined):
    """Process sex values"""
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    print('Processing Sex : ok')
    return combined

def process_pclass(combined):
    """Process passenger class"""
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    print('Processing Pclass : ok')
    return combined

def process_ticket(combined):
    """Process ticket values"""
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = [t.strip() for t in ticket]
        ticket = [t for t in ticket if not t.isdigit()]
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    
    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    
    print('Processing Ticket : ok')
    return combined

def process_family(combined):
    """Process family-related features"""
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    print('Processing family : ok')
    return combined

def main():
    """Main function to train and save the model"""
    print("Loading data...")
    combined, targets = get_combined_data()
    
    print("Feature engineering...")
    combined = get_titles(combined)
    combined = process_age(combined)
    combined = process_names(combined)
    combined = process_fares(combined)
    combined = process_embarked(combined)
    combined = process_cabin(combined)
    combined = process_sex(combined)
    combined = process_pclass(combined)
    combined = process_ticket(combined)
    combined = process_family(combined)
    
    print(f"Final dataset shape: {combined.shape}")
    
    # Split back into train and test
    train = combined.iloc[:891]
    test = combined.iloc[891:]
    
    print("Training Random Forest model...")
    
    # Feature selection using Random Forest
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    
    # Select important features
    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(test)
    
    print(f"Reduced features shape: {train_reduced.shape}")
    
    # Train final model with optimized parameters
    parameters = {
        'bootstrap': False, 
        'min_samples_leaf': 3, 
        'n_estimators': 50, 
        'min_samples_split': 10, 
        'max_features': 'sqrt', 
        'max_depth': 6
    }
    
    final_model = RandomForestClassifier(**parameters)
    final_model.fit(train_reduced, targets)
    
    # Get feature names for the reduced dataset
    feature_columns = train.columns[model.get_support()].tolist()
    
    # Save the model and feature columns
    model_data = {
        'model': final_model,
        'feature_selector': model,
        'feature_columns': feature_columns,
        'full_train_columns': train.columns.tolist()
    }
    
    with open('titanic_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully to 'titanic_model.pkl'")
    print(f"Number of features used: {len(feature_columns)}")
    
    # Test the model
    predictions = final_model.predict(test_reduced)
    print(f"Number of test predictions: {len(predictions)}")
    print(f"Sample predictions: {predictions[:10]}")

if __name__ == "__main__":
    main() 