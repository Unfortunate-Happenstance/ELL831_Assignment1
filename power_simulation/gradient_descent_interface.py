import subprocess
import numpy as np
import re
import os
import json
from itertools import product
import time
import threading
# Global Paths
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
    circuit_dir = os.path.dirname(circuit_file_name)
    subprocess.run([LTSPICE_EXECUTABLE, "-b", os.path.basename(circuit_file_name)], check=True)
    print("[DEBUG] LTspice simulation completed.")

# Function to Parse Power Consumption from .log File
def parse_power_consumption(index):
    """Extract the average power consumption from the .log file."""
    print("[DEBUG] Parsing power consumption from .log file...")
    output_log_file = f"/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_{index}.log"
    try:
        with open(output_log_file, "r",encoding="utf-16-le") as file:
            log_content = file.read()
        
        #regex to find the power value
        power_match = re.search(r"avg_power:\s*AVG\(-i\(v2\)\*v\(vdd\)\)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", log_content)
                
        if power_match:
            power = float(power_match.group(1))
            print(f"[DEBUG] Power consumption extracted: {power:.2e} W")
            return power
        else:
            print("[ERROR] Power consumption entry not found in log file.")
            raise ValueError("Power consumption not found in .log file.")
    except Exception as e:
        print(f"[ERROR] Failed to parse power consumption: {e}")
        raise

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

#Parameters
def gen_parameter_list():
    """Returns a list of all possible parameter combinations independently computed for Wp1-Wn2, Wp6-Wn8, and Wn10-Wp11."""
    print("\n[DEBUG] Starting parameter value generation...")
    
    def validate_size(size):
        """Check if the size is within the acceptable range."""
        return 320e-9 <= size <= 3000e-9
    
    step_size = 10e-9  # Sizing step size
    
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




# Optimization Loop
def optimize():
    """Run iterative optimization to minimize power consumption."""
    print("\n[DEBUG] Starting optimization loop...")
    W_range=gen_parameter_list() #get parameter ranges
    results=[]

    def run_ltspice_sim(i):
        print(f"\n[DEBUG] Iteration {i}")
        
        Wp1, Wn2, Wp6, Wn8, Wn10, Wp11 = W_range[i].values()
        
        # Update circuit file with new Wn and Wp
        update_circuit_file([Wp1, Wn2, Wp6, Wn8, Wn10, Wp11], i)  

        # Run LTspice simulation
        run_ltspice(i)
            
        # Parse power consumption from .log file
        power = parse_power_consumption(i)
            
        # Store results
        results.append((Wp1, Wn2, Wp6, Wn8, Wn10, Wp11, power))
        print(f"[DEBUG] Results stored: Wp1={Wp1:.2e}, Wn2={Wn2:.2e}, Wp6={Wp6:.2e}, Wn8={Wn8:.2e}, Wn10={Wn10:.2e}, Wp11={Wp11:.2e}, Power={power:.2e} W")    
        
        os.system(f"rm Ringamp1_{i}*")

        # print(f"[DEBUG] Files removed: Ringamp1_{i}.cir, Ringamp1_{i}.log, Ringamp1_{i}.raw")
        # os.system(f"rm Ringamp1_{i}.op.raw")
    all_results = []    
    threads = []
    for j in range(1, len(W_range)//8):
        for i in range(j*8,(j+1)*8):
            t = threading.Thread(target=run_ltspice_sim, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        if j%100 == 0:
            results_np = np.array(results)
            all_results.extend(results)
            #save results to json
            save_results_to_json(results_np, f"results/power_simulation_results_{j}.json")

            results = []
        
    results = np.array(all_results)
    
    # Find the minimum power consumption
    min_index = np.argmin(results[:, 6])
    min_par = results[min_index]
    
    print("\n[DEBUG] Optimization completed.")
    print(f"[RESULT] Optimal Parameters:",min_par)
    print(f"Wp1={int(min_par[0]*1e9):.0f}nm, Wn2={int(min_par[1]*1e9):.0f}nm, Wp6={int(min_par[2]*1e9):.0f}nm, Wn8={int(min_par[3]*1e9):.0f}nm, Wn10={int(min_par[4]*1e9):.0f}nm, Wp11={int(min_par[5]*1e9):.0f}nm")
    print(f"\n[RESULT] Minimum Power Consumption: {min_par[6]*1e06:.2f} μW")
    return results


def save_results_to_json(results_array, filename='power_simulation_results.json'):
    """
    Save numpy array of results to JSON file
    
    Parameters:
    results_array: numpy array with columns [Wp1, Wn2, Wp6, Wn8, Wn10, Wp11, power]
    filename: output JSON filename
    """
    # Convert numpy array to list of dictionaries
    results_list = []
    for row in results_array:
        results_list.append({
            'Wp1': float(row[0]),
            'Wn2': float(row[1]),
            'Wp6': float(row[2]),
            'Wn8': float(row[3]),
            'Wn10': float(row[4]),
            'Wp11': float(row[5]),
            'power': float(row[6])
        })
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(results_list, f, indent=4)

# Main Execution
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
