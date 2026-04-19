import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("Loading Cleveland Dataset from extremely local file...")
    # Load from the local file instead of downloading every time
    df = pd.read_csv('cleveland_dataset.csv', na_values='?')
    
    print("Preprocessing data...")
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert 'target' to binary (0 = no disease, 1 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print("Splitting dataset into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=32, stratify=y)
    
    print("Training High-Accuracy Machine Learning Model (Random Forest Ensemble)...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Validation Accuracy: {accuracy*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving trained model to disk as 'heart_model.pkl'...")
    joblib.dump(model, 'heart_model.pkl')
    
    print("Saving model evaluation metrics for the Streamlit App...")
    # Save metrics and a snapshot of dataframe for the app to display evaluation
    joblib.dump((accuracy, conf_matrix, X.columns.tolist(), df), 'model_metadata.pkl')
    
    print("✅ Training sequence perfectly completed.")

if __name__ == "__main__":
    main()
