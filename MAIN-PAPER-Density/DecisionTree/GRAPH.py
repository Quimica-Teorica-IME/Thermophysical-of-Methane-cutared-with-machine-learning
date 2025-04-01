import os
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(phase_name, y_exp, y_nist, y_model, save_path=None, name=None):
    plt.figure(figsize=(8, 6))

    #plt.scatter(y_exp, y_exp, color='black', label='Experimental', marker='x', s=100, linewidth=2)
    #plt.scatter(y_exp, y_nist, color='blue', label='NIST', marker='+', s=150, linewidth=2)
    #plt.scatter(y_exp, y_model, color='red', label='SVR', marker='o', facecolors='none', edgecolors='red', s=150, linewidth=2)
    
    plt.scatter(y_exp, y_exp, color='black', label='Experimental', marker='.', s=100, linewidth=2)
    plt.scatter(y_exp, y_nist, color='blue', label='NIST',  marker='o', facecolors='none', edgecolors='blue', s=150, linewidth=2)
    plt.scatter(y_exp, y_model, color='red', label='Decision Tree',  marker='+', s=150, linewidth=2)
    plt.scatter(y_exp, y_exp, color='black', label='_nolegend_', marker='.', s=50, linewidth=2)

    plt.xlabel('Experimental')
    plt.ylabel('Decision Tree & NIST')
    plt.title(phase_name)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, f'{name}.png'))
        plt.close()
    else:
        plt.show()



current_directory = os.getcwd()
save_directory = os.path.join(current_directory, 'ModelxNistxExp')
os.makedirs(save_directory, exist_ok=True)

# Look for files
files = ['liquid_data.txt', 'vapor_data.txt', 'supercritical_data.txt']

for file in files:
    file_path = os.path.join(current_directory, file)
    if not os.path.exists(file_path):
        print(f'File {file} not found in didectory {current_directory}. Skipping...')
        continue

    phase_name = file.split('_')[0]

    # Read data from file
    data = np.loadtxt(os.path.join(current_directory, file), delimiter='\t', skiprows=1)


    y_exp = data[:, 2]
    y_nist = data[:, 3]
    y_model = data[:, 4]


    # Check if file already exists
    name = file.split('.')[0]  # Use original file name as base
    counter = 1
    while os.path.exists(os.path.join(save_directory, f'{name}_{counter}.png')):
        counter += 1

    # Plot and save the comparison
    plot_comparison(phase_name, y_exp, y_nist, y_model, save_path=save_directory, name=f'{name}_{counter}')
