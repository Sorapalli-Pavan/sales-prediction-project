import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    # Load dataset
    data = pd.read_csv('backend/data/advertising.csv')  # Adjust path if needed
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model to disk
    save_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
