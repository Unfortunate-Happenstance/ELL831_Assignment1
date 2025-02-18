import math
import random
import numpy as np
import subprocess
import re 
import os
import time
import threading

##############################
# User-provided placeholders #
##############################
LTSPICE_EXECUTABLE = "/Applications/LTspice.app/Contents/MacOS/LTspice"  # MacOS path
CIRCUIT_FILE_POWER = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_power.cir"
CIRCUIT_FILE_GAIN = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_gain.cir"
OUTPUT_LOG_POWER = "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_power.log"
OUTPUT_LOG_GAIN= "/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_gain.log"

def run_ltspice(type):
    """Run LTspice simulation and generate .log and .raw files."""
    print("\n[DEBUG] Running LTspice simulation...")
    
    # Get the directory of the circuit file
    circuit_file_name = f"Ringamp1_{type}.cir"
    subprocess.run([LTSPICE_EXECUTABLE, "-b", os.path.basename(circuit_file_name)], check=True)
    print("[DEBUG] LTspice simulation completed.")

#Function to parse ac analysis data from .raw file and return gain and phase at 40kHz
def parse_ac_data():
    """Parse the AC analysis data from the .raw file."""
    print("\n[DEBUG] Parsing AC analysis data...")
    
    #Find the gain and phase at 40kHz from the .log file 
    with open(OUTPUT_LOG_GAIN, "r",encoding="utf-16-le") as file:
        log_content= file.read()
    
    # Regex pattern to extract gain and phase at 40kHz
    pattern = r"gain40khz_db:\s*20\*log10\(v\(vout\)/v\(vin\)\)=\(\s*(-?\d+\.\d+)\s*dB,\s*(-?\d+\.\d+)[°�]?\)"

    # Search for the pattern in the log file
    match = re.search(pattern, log_content, re.IGNORECASE)
    if match:
        gain_db = float(match.group(1))  # Extract Gain in dB
        phase_deg = float(match.group(2))  # Extract Phase in degrees
        print("[DEBUG] Gain and Phase at 40kHz extracted successfully.")
    else:
        print("No match found.")
    
    return gain_db, phase_deg

# Function to Parse Power Consumption from .log File
def parse_power_consumption():
    """Extract the average power consumption from the .log file."""
    print("[DEBUG] Parsing power consumption from .log file...")
    try:
        with open(OUTPUT_LOG_POWER, "r",encoding="utf-16-le") as file:
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

def update_circuit_file(new_values, type):
    """Update the .cir file with new MOSFET widths."""
    Wp1, Wn2, Wp6, Wn8,Wp11,Wn10=new_values
    print(f"\n[DEBUG] Updating circuit file with Wp1={Wp1:.2e}, Wn2={Wn2:.2e}, Wp6={Wp6:.2e}, Wn8={Wn8:.2e}, Wn10={Wn10:.2e}, Wp11={Wp11:.2e}..." )
    circuit_file=f"/Users/nakshatpandey/Third_Year/2402/ELL831/Assignment1/ELL831_Assignment1/Ringamp1_{type}.cir"
    try:
        with open(circuit_file, "r") as file:
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
        
        with open(circuit_file, "w") as file:
            file.writelines(updated_lines)
        print("[DEBUG] Circuit file updated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to update circuit file: {e}")
        raise

# def run_ltspice_simulation(W):
#     """
#     Placeholder for your custom function that:
#     1. Writes transistor widths W into the netlist.
#     2. Runs LTspice (subprocess or API).
#     3. Parses the .log/.raw to extract:
#        - Gain in dB at the target frequency (e.g. 40 kHz).
#        - Power consumption in micro-watts.
#     Returns: (gain_dB, power_uW)
#     """
#     # -- Your implementation here --
#     # Update the circuit file with the new transistor widths
#     update_circuit_file(W, "power")
#     update_circuit_file(W, "gain")
#     run_ltspice("power")
#     power_uW = parse_power_consumption()
#     run_ltspice("gain")
#     gain_dB, phase_deg = parse_ac_data()
#     print(f"[DEBUG] Gain: {gain_dB} dB, Phase: {phase_deg} degrees extracted")
#     return (gain_dB, power_uW)

def run_ltspice_simulation(W):
    """
    Run LTspice simulations for power consumption and gain in separate threads.
    """
    power_uW = 0
    gain_dB, phase_deg = 0,0 # Initialize to zero

    def run_power():
        update_circuit_file(W, "power")
        run_ltspice("power")
        nonlocal power_uW
        power_uW = parse_power_consumption()
    
    def run_gain():
        update_circuit_file(W, "gain")
        run_ltspice("gain")
        nonlocal gain_dB, phase_deg
        gain_dB, phase_deg = parse_ac_data()
    
    
    thread_power = threading.Thread(target=run_power)
    thread_gain = threading.Thread(target=run_gain)
    
    thread_power.start()
    thread_gain.start()
    
    thread_power.join()
    thread_gain.join()
    
    print(f"[DEBUG] Gain: {gain_dB} dB, Phase: {phase_deg} degrees extracted")
    return (gain_dB, power_uW)


def validate_size(size):
    """Check if the size is within the acceptable range."""
    return 320e-9 <= size <= 3000e-9

def generate_neighbor(current_solution):
    """
    Generate a neighbor solution by perturbing Wp transistor widths and maintaining ratios for Wn.
    
    current_solution: A tuple of transistor widths in order (Wp1, Wn2, Wp6, Wn8, Wp11, Wn10).
    
    Returns:
        A tuple of new transistor widths, where Wp are perturbed by ±20 nm and Wn maintain ratios,
        all values clipped to [320e-9, 3000e-9] and rounded to nearest nm.
    """
    neighbor_solution = []
    step_size = 20e-9  # 20 nm perturbation
    nm_step = 1e-9    # 1 nm precision
    
    # Define the ratios
    ratio1 = 2520e-9 / 960e-9   # Wp1/Wn2 ratio
    ratio2 = 840e-9 / 320e-9    # Wp6/Wn8 ratio
    ratio3 = 960e-9 / 320e-9    # Wp11/Wn10 ratio
    
    # Get p-transistor indices (0, 2, 4)
    p_indices = [0, 2, 4]
    ratios = [ratio1, ratio2, ratio3]
    
    # Process each p-transistor and its corresponding n-transistor
    for i, p_idx in enumerate(p_indices):
        # Perturb the p-transistor
        Wp = current_solution[p_idx]
        delta = random.uniform(-step_size, step_size)
        delta = round(delta/nm_step) * nm_step
        new_Wp = Wp + delta
        
        # Clip to allowed range
        if not validate_size(new_Wp):
            new_Wp = max(320e-9, min(new_Wp, 3000e-9))
            
        # Calculate corresponding n-transistor size maintaining ratio
        new_Wn = round(new_Wp / ratios[i] / nm_step) * nm_step
        
        # Clip n-transistor size if needed
        if not validate_size(new_Wn):
            new_Wn = max(320e-9, min(new_Wn, 3000e-9))
            # Recalculate Wp to maintain exact ratio if Wn was clipped
            new_Wp = round(new_Wn * ratios[i] / nm_step) * nm_step
        
        neighbor_solution.extend([new_Wp, new_Wn])
    
    print("[DEBUG] Generated neighbor solution:")
    return tuple(neighbor_solution)

#####################################
# Cost function with normalization  #
#####################################

def cost_function(W, G_ref=35.0, P_ref=40.0):
    """Calculate the cost function for given transistor widths."""
    print("\n[DEBUG] Evaluating cost function...")
    gain_dB, power_uW = run_ltspice_simulation(W)
    if gain_dB is None or power_uW is None:
        print("[ERROR] Invalid gain or power values. Skipping cost evaluation.")
        return float('inf')
    termA = -(gain_dB / G_ref)
    termB = power_uW / P_ref
    cost = 0.5 * termA + 0.5 * termB
    print(f"[INFO] Computed cost: {cost}")
    # returns the cost, the gain and the power
    return cost


########################################
# Simulated Annealing High-level Logic #
########################################
#SA
def simulated_annealing(initial_solution, T_init=1.0, alpha=0.95, max_iter=100):
    """Perform simulated annealing optimization."""
    print("\n[INFO] Starting simulated annealing...")
    current_solution = initial_solution
    current_cost = cost_function(current_solution)
    best_solution, best_cost = current_solution, current_cost
    T = T_init
    
    for i in range(max_iter):
        print(f"\n[INFO] Iteration {i+1}/{max_iter}, Temperature: {T:.5f}")
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = cost_function(neighbor)
        cost_diff = neighbor_cost - current_cost
        
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / T):
            current_solution, current_cost = neighbor, neighbor_cost
            print("[DEBUG] Accepted new solution.")
            
        if current_cost < best_cost:
            best_solution, best_cost = current_solution, current_cost
            print("[INFO] Found new best solution.")
        
        T *= alpha
    
    print(f"\n[INFO] Optimization completed. Best cost: {best_cost}, Best solution: {best_solution}")
    return best_solution, best_cost


#Main Execution
if __name__ == "__main__":
    a=time.time()
    print("[DEBUG] Script started.")
    # Initial solution (original parameters)
    #list from * original ckt parameters Wp1=2520n Wn2=960n Wp6=840n Wn8=320n Wp11=960n Wn10=320n
    initial_solution = (2520e-9, 960e-9, 840e-9, 320e-9, 960e-9, 320e-9)
    simulated_annealing(initial_solution)
    b=time.time()
    elapsed_time=b-a
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Execution time: {formatted_time}")