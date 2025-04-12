import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def train_breast_cancer():
    # Load data
    df = pd.read_csv(r"e:\semester 4\cancer_prediction\Multiple-Cancers-Classification-master - backup\breast_cancer.csv")

    
    # Clean and prepare data
    df.columns = df.columns.str.strip()
    y = df['Status'].map({'Alive': 0, 'Dead': 1})
    
    # Feature selection and renaming
    X = df[[
        'Age', 'Race', 'Marital Status', 'T Stage', 'N Stage',
        '6th Stage', 'differentiate', 'Grade', 'A Stage',
        'Tumor Size', 'Estrogen Status', 'Progesterone Status',
        'Regional Node Examined', 'Regional Node Positive', 'Survival Months'
    ]].rename(columns={
        'Marital Status': 'Marital_Status',
        'T Stage': 'T_Stage',
        'N Stage': 'N_Stage',
        '6th Stage': 'Stage_6th',
        'A Stage': 'A_Stage',
        'Regional Node Positive': 'Regional_Node_Positive'
    })

    # Define feature types
    categorical_features = [
        'Race', 'Marital_Status', 'T_Stage', 'N_Stage',
        'Stage_6th', 'differentiate', 'Estrogen Status',
        'Progesterone Status', 'Grade', 'A_Stage'
    ]
    
    numerical_features = [
        'Age', 'Tumor Size', 'Regional Node Examined',
        'Regional_Node_Positive', 'Survival Months'
    ]

    # Create pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, r"e:\semester 4\cancer_prediction\Multiple-Cancers-Classification-master - backup\models\breast_model.pkl")

    print("\nModel saved successfully!")

    # Demo prediction
    print("\nSample Prediction:")
    sample = X_test.iloc[0:1]  # Get first test sample
    print("\nInput Features:")
    print(sample)
    prediction = model.predict(sample)
    proba = model.predict_proba(sample)
    print(f"\nPrediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
    print(f"Confidence: {max(proba[0]):.2%}")

if __name__ == '__main__':
    train_breast_cancer()