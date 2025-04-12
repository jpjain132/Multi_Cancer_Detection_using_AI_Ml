from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all models
models = {
    'breast': joblib.load(r"e:\semester 4\cancer_prediction\Multiple-Cancers-Classification-master - backup\models\breast_model.pkl"),
    # 'prostate': joblib.load('models/prostate_model.pkl'),
    # 'lung': joblib.load('models/lung_model.pkl'),
    # 'colorectal': joblib.load('models/colorectal_model.pkl'),
    # 'pancreatic': joblib.load('models/pancreatic_model.pkl')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cancer_type = request.form['cancer_type']
    result = "Error in prediction"
    
    if cancer_type == 'breast':
        try:
            # Map form fields to dataframe columns exactly
            features = {
                'Age': float(request.form['Age']),
                'Race': request.form['Race'],
                'Marital_Status': request.form['Marital_Status'],  # Match column name with space
                'T_Stage': request.form['T_Stage'],
                'N_Stage': request.form['N_Stage'],
                'Stage_6th': request.form['Stage_6th'],
                'differentiate': request.form['differentiate'],
                'Grade': request.form['Grade'],
                'A_Stage': request.form['A_Stage'],
                'Tumor Size': float(request.form['Tumor_Size']),
                'Estrogen Status': request.form['Estrogen_Status'],
                'Progesterone Status': request.form['Progesterone_Status'],
                'Regional Node Examined': float(request.form['Regional Node Examined']),
                'Regional_Node_Positive': float(request.form['Regional_Node_Positive']),
                'Survival Months': float(request.form['Survival_Months'])
            }
            
            # Create DataFrame with exact column order
            input_df = pd.DataFrame([features])
            
            # Get prediction
            model = models['breast']
            prediction = model.predict(input_df)[0]
            result = 'High Risk' if prediction == 1 else 'Low Risk'
            
        except KeyError as e:
            return f"Missing form field: {str(e)}", 400
        except ValueError as e:
            return f"Invalid input: {str(e)}", 400
    
    return render_template('result.html', 
                        prediction=result,
                        cancer_type=cancer_type.capitalize())

if __name__ == '__main__':
    app.run(debug=True)