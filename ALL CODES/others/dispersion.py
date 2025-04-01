import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

properties = [
    "Density (kg/m3)", "Volume (m3/kg)", "Enthalpy (kJ/mol)",
    "Cv (J/mol*K)", "Cp (J/mol*K)", "Sound Spd (m/s)",
    "Joule-Thomson (K/MPa)", "Viscosity (uPa*s)"
]

if not os.path.exists('plots'):
    os.makedirs('plots')

phases = ["liquid", "vapor", "supercritical"]

base_path = os.getcwd()

for property in properties:
    property_name = property.split(' (')[0]
    folder_path = os.path.join(base_path, property_name)

    if not os.path.isdir(folder_path) or property_name == 'plots':
        continue

    for phase in phases:
        file_path = os.path.join(folder_path, f"{phase}_data.txt")
        if not os.path.isfile(file_path):
            continue 

        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_csv(file_path, sep=',') 

        expected_columns = ["Temperature (K)", "Pressure (MPa)", property, "Reference"]
        if not all(col in df.columns for col in expected_columns):
            print(f"File {file_path} does not have the expected columns.")
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        df["Reference"] = df["Reference"].str.strip().str.lower()
        references = df["Reference"].unique()
        palette = sns.color_palette("hsv", len(references))
        colors = dict(zip(references, palette))

        for ref in references:
            sub_df = df[df["Reference"] == ref]
            ax.scatter(
                sub_df["Temperature (K)"],
                sub_df["Pressure (MPa)"],
                sub_df[property],
                label=ref,
                color=colors[ref]
            )

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_zlabel(property)
        ax.set_title(f"{property_name} - {phase}")

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.1, 0.8),
            title="Reference",
            fontsize='medium'
        )

        save_path = os.path.join(base_path, "plots", f"{property_name}_{phase}_dispersion.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()
