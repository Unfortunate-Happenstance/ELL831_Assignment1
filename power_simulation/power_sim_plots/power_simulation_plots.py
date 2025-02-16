import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


def load_results_from_json(filename='transistor_results.json'):
    """Load results from JSON file and convert to appropriate format"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert list of dictionaries to numpy array
    results_array = np.array([[d['Wp1'], d['Wn2'], d['Wp6'], 
                              d['Wn8'], d['Wn10'], d['Wp11'], 
                              d['power']] for d in data])
    return results_array


def plot_transistor_analysis(data):
    """
    Create visualizations to analyze the effect of transistor sizing on power.
    """
    plt.style.use('seaborn')
    
    # Extract independent variables (transistor sets)
    wp1_wn2 = data['Wp1']  # Same as Wn2 * fixed ratio
    wp6_wn8 = data['Wp6']  # Same as Wn8 * fixed ratio
    wn10_wp11 = data['Wp11']  # Same as Wn10 * fixed ratio
    power = data['power']
    
    # Figure 1: Scatter Plots of Individual Transistor Sizing vs Power
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].scatter(wp1_wn2, power, alpha=0.5, color='r')
    axes[0].set_xlabel('Wp1-Wn2 Sizing (nm)')
    axes[0].set_ylabel('Power')
    axes[0].set_title('Wp1-Wn2 vs Power')
    
    axes[1].scatter(wp6_wn8, power, alpha=0.5, color='g')
    axes[1].set_xlabel('Wp6-Wn8 Sizing (nm)')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Wp6-Wn8 vs Power')
    
    axes[2].scatter(wn10_wp11, power, alpha=0.5, color='b')
    axes[2].set_xlabel('Wn10-Wp11 Sizing (nm)')
    axes[2].set_ylabel('Power')
    axes[2].set_title('Wn10-Wp11 vs Power')
    
    fig1.tight_layout()
    
    # Figure 2: 3D Scatter Plot of Transistor Sizing vs Power
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    scatter = ax.scatter(wp1_wn2, wp6_wn8, wn10_wp11, c=power, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Wp1-Wn2 Sizing (nm)')
    ax.set_ylabel('Wp6-Wn8 Sizing (nm)')
    ax.set_zlabel('Wn10-Wp11 Sizing (nm)')
    ax.set_title('3D Transistor Sizing vs Power')
    plt.colorbar(scatter, label='Power')
    
    # Figure 3: Correlation Bar Chart
    fig3, ax = plt.subplots(figsize=(8, 6))
    correlations = [
        np.corrcoef(wp1_wn2, power)[0, 1],
        np.corrcoef(wp6_wn8, power)[0, 1],
        np.corrcoef(wn10_wp11, power)[0, 1]
    ]
    labels = ['Wp1-Wn2', 'Wp6-Wn8', 'Wn10-Wp11']
    ax.bar(labels, correlations, color=['r', 'g', 'b'])
    ax.set_ylabel('Correlation with Power')
    ax.set_title('Correlation of Transistor Sizing with Power')
    
    # Figure 4: Regression Analysis
    fig4, ax = plt.subplots(figsize=(8, 6))
    X = np.column_stack((wp1_wn2, wp6_wn8, wn10_wp11))
    model = LinearRegression()
    model.fit(X, power)
    predicted_power = model.predict(X)
    
    ax.scatter(power, predicted_power, alpha=0.5)
    ax.plot([min(power), max(power)], [min(power), max(power)], linestyle='--', color='k')
    ax.set_xlabel('Actual Power')
    ax.set_ylabel('Predicted Power')
    ax.set_title('Regression Fit (Sensitivity Analysis)')
    
    print("Regression Coefficients:")
    for label, coef in zip(labels, model.coef_):
        print(f"{label}: {coef:.5f}")
    
    fig1.savefig('transistor_sizing_vs_power.png', dpi=300, bbox_inches='tight')
    fig2.savefig('3D_transistor_sizing.png', dpi=300, bbox_inches='tight')
    fig3.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    fig4.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
    
    return fig1, fig2, fig3, fig4


def main():
    # Load data from JSON
    results_array = load_results_from_json('power_sim_old.json')
    
    # Convert to dictionary format for plotting
    data = {
        'Wp1': results_array[:, 0],
        'Wn2': results_array[:, 1],
        'Wp6': results_array[:, 2],
        'Wn8': results_array[:, 3],
        'Wn10': results_array[:, 4],
        'Wp11': results_array[:, 5],
        'power': results_array[:, 6]
    }
    
    # Create and show plots
    fig1, fig2, fig3, fig4 = plot_transistor_analysis(data)
    plt.show()
    

if __name__ == "__main__":
    main()
