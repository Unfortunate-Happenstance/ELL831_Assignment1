import numpy as np
import subprocess
from PyLTSpice import RawRead
import matplotlib.pyplot as plt
import re 
import os
import json
import pickle
from itertools import product
import threading
import time

LTSPICE_EXECUTABLE = "/Applications/LTspice.app/Contents/MacOS/LTspice"  # MacOS path
CIRCUIT_FILE = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1.cir"
OUTPUT_LOG = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1.log"
OUTPUT_RAW = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1.op.raw"

# Function to Run LTSpice Simulation
def run_ltspice(index):
    """Run LTspice simulation and generate .log and .raw files."""
    print("\n[DEBUG] Running LTspice simulation...")
    
    # Get the directory of the circuit file
    circuit_file_name = f"Ringamp1_{index}.cir"
    subprocess.run([LTSPICE_EXECUTABLE, "-b", os.path.basename(circuit_file_name)], check=True)
    print("[DEBUG] LTspice simulation completed.")

def gen_parameter_list():
    """Returns a list of all possible parameter combinations independently computed for Wp1-Wn2, Wp6-Wn8, and Wn10-Wp11."""
    print("\n[DEBUG] Starting parameter value generation...")
    
    def validate_size(size):
        """Check if the size is within the acceptable range."""
        return 320e-9 <= size <= 3000e-9
    
    step_size = 20e-9  # Sizing step size
    
    # Define ratios
    ratio1 = 2520e-9 / 960e-9  # Wp1/Wn2 ratio
    ratio2 = 840e-9 / 320e-9   # Wp6/Wn8 ratio
    ratio3 = 960e-9 / 320e-9   # Wp11/Wn10 ratio
    
    # Generate independent valid sizes for each set
    base_sizes = np.arange(320e-9, 3000e-9 + step_size, step_size)
    
    valid_set1 = [(round((size * ratio1) * 1e9) * 1e-9, size) for size in base_sizes if validate_size(size * ratio1)]
    valid_set2 = [(round((size * ratio2) * 1e9) * 1e-9, size) for size in base_sizes if validate_size(size * ratio2)]
    valid_set3 = [(size, round((size * ratio3) * 1e9) * 1e-9) for size in base_sizes if validate_size(size * ratio3)]
    
    # Compute Cartesian product of all independent sets
    valid_combinations = []
    for (Wp1, Wn2), (Wp6, Wn8), (Wn10, Wp11) in product(valid_set1, valid_set2, valid_set3):
        valid_combinations.append({
            'Wp1': Wp1, 'Wn2': Wn2,
            'Wp6': Wp6, 'Wn8': Wn8,
            'Wn10': Wn10, 'Wp11': Wp11
        })
    
    print(f"\n[DEBUG] Generated {len(valid_combinations)} parameter combinations.")
    return valid_combinations

# def gen_parameter_list():
#     '''returns a list of 68 possible combinations of Wp1, Wn2, Wp6, Wn8, Wn10, Wp11'''
#     print("\n[DEBUG] Starting parameter value generation...")
#     def validate_sizes(size):
#         return 320e-9 <= size <= 3000e-9
#     step_size = 10e-9

#     ratio1 = 2520e-9 / 960e-9  # Wp1/Wn2 ratio
#     ratio2 = 840e-9 / 320e-9   # Wp6/Wn8 ratio
#     ratio3 = 960e-9 / 320e-9   # Wp11/Wn10 ratio

#     valid_combinations=[]
#     base_sizes = np.arange(320e-9, 3000e-9 + step_size, step_size)

#     for base_size in base_sizes:
#         # Calculate paired sizes based on ratios
#         combinations = [
#             # (Wp1, Wn2)
#             (base_size * ratio1, base_size),
#             # (Wp6, Wn8)
#             (base_size * ratio2, base_size),
#             # (Wn10, Wp11)
#             (base_size, base_size * ratio3)
#         ]
    
#         # Check if all sizes are valid
#         valid=True
#         for size1,size2 in combinations:
#             if not (validate_sizes(size1) and validate_sizes(size2)):
#                 valid=False
#                 break
#         if valid:
#             formatted_combo = {
#                 'Wp1': round(combinations[0][0]*1e9)*1e-9 ,
#                 'Wn2': round(combinations[0][1]*1e9)*1e-9 ,
#                 'Wp6': round(combinations[1][0]*1e9)*1e-9 ,
#                 'Wn8': round(combinations[1][1]*1e9)*1e-9 ,
#                 'Wn10':round(combinations[2][0]*1e9)*1e-9 ,
#                 'Wp11':round(combinations[2][1]*1e9)*1e-9 
#             }
#             valid_combinations.append(formatted_combo)
#     print("\n[DEBUG] parameters generated")
#     return valid_combinations


#Function to parse ac analysis data from .raw file and return gain and phase at 40kHz
def parse_ac_data(index):
    """Parse the AC analysis data from the .raw file."""
    print("\n[DEBUG] Parsing AC analysis data...")
    
    output_raw_file=f"/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_{index}.raw"
    output_log_file=f"/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_{index}.log"
    # Read the .raw file
    data = RawRead(output_raw_file)
    sim_result=[]
    # Extract frequency, gain, and phase data
    # gain = data.get_trace("V(vout)/V(vin)")   # Gain (V/V)
    vout=data.get_trace("V(vout)")
    print("[DEBUG] Extracting frequency data...")
    frequency = data.get_axis()  # Frequency (Hz)
    sim_result.append(vout) 
    sim_result.append(frequency)    

    #Find the gain and phase at 40kHz from the .log file 
    with open(output_log_file, "r",encoding="utf-16-le") as file:
        log_content= file.read()
    
    # Regex pattern to extract gain and phase at 40kHz
    pattern = r"gain40khz_db:\s*20\*log10\(v\(vout\)/v\(vin\)\)=\(\s*(-?\d+\.\d+)\s*dB,\s*(-?\d+\.\d+)[°�]?\)"

    # Search for the pattern in the log file
    match = re.search(pattern, log_content, re.IGNORECASE)
    if match:
        gain_db = float(match.group(1))  # Extract Gain in dB
        sim_result.append(gain_db)
        phase_deg = float(match.group(2))  # Extract Phase in degrees
        sim_result.append(phase_deg)
        print("[DEBUG] Gain and Phase at 40kHz extracted successfully.")
    else:
        print("No match found.")
    
    return sim_result


# Function to Update .cir File with New Parameter Values
def update_circuit_file(new_values, index):
    """Update the .cir file with new MOSFET widths."""
    Wp1, Wn2, Wp6, Wn8,Wn10,Wp11=new_values
    print(f"\n[DEBUG] Updating circuit file with Wp1={Wp1:.2e}, Wn2={Wn2:.2e}, Wp6={Wp6:.2e}, Wn8={Wn8:.2e}, Wn10={Wn10:.2e}, Wp11={Wp11:.2e}..." )
    try:
        with open(CIRCUIT_FILE, "r") as file:
            lines = file.readlines()
        
        # Modify relevant parameters in .cir file
        updated_lines = []
        for line in lines:
            if ".param" in line:
                # Update the entire .param line to include Wn and Wp
                updated_lines.append(f".param Wp1={Wp1:.2e}, Wn2={Wn2:.2e}, Wp6={Wp6:.2e}, Wn8={Wn8:.2e}, Wn10={Wn10:.2e}, Wp11={Wp11:.2e} ; define widths \n")
                print(f"[DEBUG] Updated Wp1 to {Wp1:.2e}, Wn2 to {Wn2:.2e}, Wp6 to {Wp6:.2e}, Wn8 to {Wn8:.2e}, Wn10 to {Wn10:.2e}, Wp11 to {Wp11:.2e}")
            else:
                updated_lines.append(line)
        
        new_circuit_file = f"/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_{index}.cir"
        with open(new_circuit_file, "w") as file:
            file.writelines(updated_lines)
        print("[DEBUG] Circuit file updated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to update circuit file: {e}")
        raise
        

#Optimisation loop
def optimize():
    '''Run iterative loop to generate parameter combinations, update circuit file, run simulation, and parse results.'''
    # Generate parameter combinations
    W_range = gen_parameter_list()
    print("[DEBUG] Starting optimization loop...")
    results=[]

    def run_ltspice_sim(i):
        print(f"\n[DEBUG] Iteration {i}")
        
        Wp1, Wn2, Wp6, Wn8, Wn10, Wp11 = W_range[i].values()
        
        # Update circuit file with new Wn and Wp
        update_circuit_file([Wp1, Wn2, Wp6, Wn8, Wn10, Wp11], i)  

        # Run LTspice simulation
        run_ltspice(i)

        # Parse AC analysis data
        sim_result = parse_ac_data(i)
        results.append((Wp1, Wn2, Wp6, Wn8, Wn10, Wp11, sim_result[2], sim_result[3], pickle.dumps(sim_result[0]).hex(), pickle.dumps(sim_result[1]).hex()))
        print(f"[DEBUG] Results stored: Wp1={Wp1:.2e}, Wn2={Wn2:.2e}, Wp6={Wp6:.2e}, Wn8={Wn8:.2e}, Wn10={Wn10:.2e}, Wp11={Wp11:.2e}, Gain={sim_result[2]} dB, Phase={sim_result[3]} deg")    
        os.system(f"rm Ringamp1_{i}*")
    

    all_results = []  
    threads = []
    for j in range(20, len(W_range)//40):
        for i in range(j*40,(j+1)*40,5):
            
            t = threading.Thread(target=run_ltspice_sim, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        if j%100 == 0:
            all_results.extend(results)
            #save results to json
            save_results_to_json(results, f"results/gain_simulation_results_{j}.json")
            results=[]

    gain_values = [entry[6] for entry in all_results]  # gain magnitude values
    max_gain_index = max(range(len(gain_values)), key=lambda i: gain_values[i])

    print("[RESULT] Max gain element:", all_results[max_gain_index])
    return all_results


def save_results_to_json(results_full, filename="gain_simulation_results.json"):
    """Save the simulation results to a JSON file."""
    print(f"\n[DEBUG] Saving results to {filename}...")
    results_list = []
    for row in results_full:
        results_list.append({
            'Wp1': float(row[0]),
            'Wn2': float(row[1]),
            'Wp6': float(row[2]),
            'Wn8': float(row[3]),
            'Wn10': float(row[4]),
            'Wp11': float(row[5]),
            'gain_mag': float(row[6]),
            'phase_deg': float(row[7]),
            'Vout': row[8],
            'Frequency': row[9]
            })
            # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(results_list, f, indent=4)


#Main Execution
if __name__ == "__main__":
    a=time.time()
    print("[DEBUG] Script started.")
    results=optimize()
    save_results_to_json(results)
    print("[DEBUG] results dumped")
    print("[DEBUG] Script finished.")
    b=time.time()
    elapsed_time=b-a
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Execution time: {formatted_time}")

