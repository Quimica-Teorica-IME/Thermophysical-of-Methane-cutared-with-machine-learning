import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

root_dir = os.getcwd()
global_plots_dir = os.path.join(root_dir, 'plots')
os.makedirs(global_plots_dir, exist_ok=True)

# Loop through each property
for property_name in os.listdir(root_dir):
    if property_name == 'plots':
        continue 

    property_path = os.path.join(root_dir, property_name)
    
    if os.path.isdir(property_path):
        print(f"\nProcessing property: {property_name}...")

        phases = ["liquid", "vapor", "supercritical"]

        for phase in phases:
            original_filename = f"{phase}_data.txt"
            file_path = os.path.join(property_path, original_filename)

            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue

            df = pd.read_csv(file_path, sep='\t')

            if df.empty or df[['Temperature (K)', 'Pressure (MPa)']].dropna().empty:
                print(f"Skipping {property_name} - {phase} (no valid data)")
                continue

            X = df[['Temperature (K)', 'Pressure (MPa)']].dropna()

            if len(X) == 0:
                print(f"Skipping {property_name} - {phase} (no valid data points)")
                continue

            # Estimate contamination using LOF
            lof = LocalOutlierFactor(n_neighbors=20)
            try:
                y_pred = lof.fit_predict(X)
            except ValueError as e:
                print(f"LOF failed on {property_name} - {phase}: {e}")
                continue

            estimated_contamination = np.sum(y_pred == -1) / len(X)
            estimated_contamination = max(estimated_contamination, 1e-3)

            # Isolation Forest
            contamination = 'auto' if estimated_contamination == 0.0 else estimated_contamination

            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=100,
                max_samples=0.8,
                random_state=42
            )
            outlier_pred = iso_forest.fit_predict(X)
            df['Outlier'] = outlier_pred  # -1 = outlier, 1 = normal

            # Separate data
            df_outliers = df[df['Outlier'] == -1]
            df_normal = df[df['Outlier'] == 1]
            normal_percentage = round(len(df_normal) / len(df) * 100)
            outliers_percentage = round(len(df_outliers) / len(df) * 100)

            plt.figure(figsize=(10, 6))
            plt.scatter(df_normal['Temperature (K)'], df_normal['Pressure (MPa)'],
                        label=f'Normal - {normal_percentage}% ({len(df_normal)} points)', alpha=0.5)
            plt.scatter(df_outliers['Temperature (K)'], df_outliers['Pressure (MPa)'],
                        color='r', label=f'Outliers - {outliers_percentage}% ({len(df_outliers)} points)', alpha=0.7)

            plt.xlabel('Temperature (K)', fontsize=16)  # Axis label size
            plt.ylabel('Pressure (MPa)', fontsize=16)   # Axis label size
            plt.legend(fontsize=12)  # Legend font size
            plt.title(f'{property_name} - {phase.capitalize()} Phase - Outlier Detection', fontsize=16)  # Title font size
            plt.tick_params(axis='both', which='major', labelsize=16)  # Axis ticks font size

            plot_filename = f"{property_name}_{phase}_outliers_plot.png"
            plot_path = os.path.join(global_plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            new_filename = f"{property_name}_{phase}_data.txt"
            new_file_path = os.path.join(property_path, new_filename)
            df.to_csv(new_file_path, sep='\t', index=False)
