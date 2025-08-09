from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd
import nest_asyncio
from pyngrok import ngrok
import uvicorn

model = joblib.load("data/ml/model3.pkl")
scaler = joblib.load("data/ml/scaler.pkl")
feature_columns = joblib.load("data/ml/columns.pkl")

app = FastAPI()

class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

def preprocess_input_data(input_features):
    input_df = pd.DataFrame([input_features], columns=[
        'cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
        'num_executed_instructions', 'execution_time', 'energy_efficiency', 
        'task_type', 'task_priority'
    ])
    input_df['task_status'] = 'completed'
    input_df['cpu_mem_product'] = input_df['cpu_usage'] * input_df['memory_usage']
    input_df['cpu_network_ratio'] = input_df['cpu_usage'] / (input_df['network_traffic'] + 1e-5)
    input_df['power_per_cpu'] = input_df['power_consumption'] / (input_df['cpu_usage'] + 1e-5)
    input_df['execution_per_instruction'] = input_df['execution_time'] / (input_df['num_executed_instructions'] + 1)
    scaling_cols = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
                   'num_executed_instructions', 'execution_time', 'energy_efficiency',
                   'cpu_mem_product', 'cpu_network_ratio', 'power_per_cpu', 'execution_per_instruction']
    input_df[scaling_cols] = scaler.transform(input_df[scaling_cols])
    input_df = pd.get_dummies(input_df, columns=['task_type', 'task_priority', 'task_status'], drop_first=True)

    if 'energy_efficiency' in input_df.columns:
        input_df = input_df.drop('energy_efficiency', axis=1)
    
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    return input_df

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        if 'features' not in data:
            return {"error": "Missing 'features' in request"}

        if len(data['features']) != 9:
            return {"error": f"Expected 9 features, got {len(data['features'])}."}

        processed_df = preprocess_input_data(data['features'])

        prediction = model.predict(processed_df)

        if isinstance(prediction[0], str):
            predicted_class = prediction[0]
            prediction_idx = {v: k for k, v in class_mapping.items()}[predicted_class]
        else:
            prediction_idx = int(prediction[0])
            predicted_class = class_mapping.get(prediction_idx, 'Unknown')

        return {
            "prediction": prediction_idx,
            "predicted_class": predicted_class,
            "confidence": "Model prediction successful"
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)