# code to read power_sim_old.json file and find the minimum power consumption
#     min_power = np.min(results[:, -1])
#     min_power_index = np.argmin(results[:, -1])
#     min_power_params = results[min_power_index, :-1]
#     print(f"[RESULT] Minimum Power Consumption: {min_power:.2f} μW")
#     print(f"[RESULT] Optimal Parameters: {min_power_params}")
import json
import numpy as np 

def load_results_from_json(filename='transistor_results.json'):
    """Load results from JSON file and convert to appropriate format"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert list of dictionaries to numpy array
    results_array = np.array([[d['Wp1'], d['Wn2'], d['Wp6'], 
                              d['Wn8'], d['Wn10'], d['Wp11'], 
                              d['power']] for d in data])
    return results_array

results=load_results_from_json('power_simulation_results.json')
min_power_index = np.argmin(results[:, -1])
min_power_params = results[min_power_index]
min_power=min_power_params[-1]*1e6
print(f"[RESULT] Minimum Power Consumption: {min_power:.2f} μW")
print(f"[RESULT] Optimal Parameters: {min_power_params}")

