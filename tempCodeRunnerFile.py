
    # Convert YES/NO to 1/0
    y = y.map({'YES': 1, 'NO': 0})
    
    # Convert categorical features
    X['GENDER'] = X['GENDER'].map({'M': 1, 'F': 0})
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Train-test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model training
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    
    # Save artifacts
    joblib.dump(model, 'models/lung_model.pkl')
    joblib.dump(scaler, 'scalers/lung_scaler.pkl')
    
    return accuracy_score(y_test, model.predict(X_test))

# Repeat similar functions for other cancer types...

if __name__ == '__main__':
    acc_breast = train_breast_cancer()
    acc_lung = train_lung_cancer()
    print(f"Breast Cancer Model Accuracy: {acc_breast:.2f}")
    print(f"Lung Cancer Model Accuracy: {acc_lung:.2f}")