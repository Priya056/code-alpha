import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv('titanic_data.csv')
data = preprocess_data(data)  # Assume a function for preprocessing

# Select features and target
features = data[['Pclass', 'Age', 'Sex']]
target = data['Survived']

# Convert categorical data to numerical
features['Sex'] = features['Sex'].map({'male': 0, 'female': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

# Function to predict survival
def predict_survival(pclass, age, sex):
    return model.predict([[pclass, age, sex]])[0]

# Example usage
survival_chance = predict_survival(1, 25, 1)  # 1st class, 25 years old, female
print(f'Survival Chance: {"Yes" if survival_chance else "No"}')
