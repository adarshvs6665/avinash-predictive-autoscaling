import json
import requests

with open('./data/json/metrics.json', 'r') as f:
    raw_data = json.load(f)

features = [
    raw_data['cpu_usage'],
    raw_data['memory_usage'],
    raw_data['network_traffic'],
    raw_data['power_consumption'],
    raw_data['num_executed_instructions'],
    raw_data['execution_time'],
    raw_data['energy_efficiency'],
    raw_data['task_type'],
    raw_data['task_priority']
]

payload = {"features": features}

response = requests.post("http://localhost:8000/predict", json=payload)

print("Response :", response.json())
