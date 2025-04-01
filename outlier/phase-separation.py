import os
import math
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    r2_score,
    mean_absolute_percentage_error
)

# ********************** FUNCTION DEFINITIONS **********************
def check_and_get_filename(directory, base_name, extension):
    """
    Check if the specific file already exists and create a new name if necessary.
    """
    count = 0
    while True:
        file_name = f"{base_name}{'' if count == 0 else f'_{count}'}.{extension}"
        full_path = os.path.join(directory, file_name)
        if not os.path.exists(full_path):
            return full_path
        count += 1

def save_to_file_and_print(message, file):
    """Prints a message and appends it to a file."""
    try:
        print(message)
        with open(file, 'a') as f:
            f.write(message)
    except Exception as e:
        print(f"An error occurred while saving the message to the file: {e}")


def calculate_metrics(phase_name, y_true, y_pred, metrics_to_print=None, file=None):
    '''
    Calculates metrics.
    If no specific metrics are requested (metrics_to_print=None), it calculates
    and includes all available metrics in the message. Otherwise, it only
    calculates the necessary ones.
    The message is saved in a file if requested (file = 'name')
    '''
    if metrics_to_print is None:
        metrics_to_print = ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2']

    metric_functions = {
        'MAPE': lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'RMSE': lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score
    }

    output_message = f'\nMetrics for {phase_name}:\n'


def separate_by_phase(df, decision_tree_model, state_mapping):
    """
    Predicts the phase using the decision tree model and returns three DataFrames:
    one for each phase (liquid, vapor, supercritical), without modifying the original DataFrame.
    """
    X = df[["Temperature (K)", "Pressure (MPa)"]].values
    state_pred = decision_tree_model.predict(X)
    
    # Reverse mapping: number -> phase name
    inv_mapping = {v: k for k, v in state_mapping.items()}
    
    # Create masks for each phase
    mask_liq = state_pred == state_mapping["liquid"]
    mask_vap = state_pred == state_mapping["vapor"]
    mask_sup = state_pred == state_mapping["supercritical"]
    
    df_liquid = df[mask_liq].copy()
    df_vapor = df[mask_vap].copy()
    df_supercritical = df[mask_sup].copy()
    
    return df_liquid, df_vapor, df_supercritical


# ********************** PARAMETER DEFINITIONS **********************
current_directory = os.getcwd()



# ********************** DATA ACQUISITION **********************
                     #*****  NIST  *****
            # **** Temperature, Pressure, Phase ****
'''
In this section, data from NIST is acquired for Temperature, Pressure, and the
corresponding physical state. These data are solely used to train a decision
tree to determine the phase in which the fluid exists at a given pressure and
temperature.
'''

X_NIST = []
state_column = []

df_NIST = pd.read_csv('NIST.txt', delimiter='\t')
X_NIST_file = df_NIST[["Temperature (K)","Pressure (MPa)"]].values
state_column_file = df_NIST.iloc[:, -1]  # Assuming the last column indicates the state

X_NIST.extend(X_NIST_file)
state_column.extend(state_column_file)

X_NIST = np.array(X_NIST)


# ********************** DECISION TREE **********************
DT_file = check_and_get_filename(current_directory, "DecisionTree_Physical-State-Determination", "log")
## Mapping of physical states. Associate them with numbers (0, 1, 2) for use in the decision tree
state_mapping = {'liquid': 0, 'vapor': 1, 'supercritical': 2}
state_mapped = [state_mapping[state] for state in state_column]

# Spliting the data into training (70%) and testing (30%) sets (better than 80-20)
X_train, X_test, state_train, state_test = train_test_split(X_NIST, state_mapped, test_size=0.3, random_state=42)

log_info = f"Decision tree to predict physical states based on temperature and pressure inputs.\nNumber of data: {len(X_NIST)}\nNumber of train data: {len(X_train)}\nNumber of test data: {len(X_test)}"
save_to_file_and_print(log_info, DT_file)

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, state_train)

state_pred = decision_tree.predict(X_test)

accuracy_tree = accuracy_score(state_test, state_pred)

save_to_file_and_print(f"\nAccuracy of Decision Tree model: {accuracy_tree * 100:.3f}%", DT_file)
save_to_file_and_print(classification_report(state_test, state_pred), DT_file)

# Metrics - Decision Tree Evaluation
save_to_file_and_print("\n\n > Decision Tree Evaluation <", DT_file)
calculate_metrics('Decision Tree', state_test, state_pred, ['RMSE', 'R2'], DT_file)


# ********************** PHASE SEPARATION **********************
for root, dirs, files in os.walk(current_directory):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
            try:
                df_csv = pd.read_csv(csv_path)

                if "Temperature (K)" in df_csv.columns and "Pressure (MPa)" in df_csv.columns:
                    df_liq, df_vap, df_sup = separate_by_phase(df_csv, decision_tree, state_mapping)

                    df_liq.to_csv(os.path.join(root, "liquid_data.txt"), index=False, sep='\t')
                    df_vap.to_csv(os.path.join(root, "vapor_data.txt"), index=False, sep='\t')
                    df_sup.to_csv(os.path.join(root, "supercritical_data.txt"), index=False, sep='\t')


            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
