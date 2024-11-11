'''
Script used to export NIST data - specific values from Experimental
'''

# ===================================================
#                INITIAL DEFINITIONS
# ===================================================
import os
import requests
import numpy as np
import pandas as pd


#files and directories
current_directory = os.getcwd()
#from google.colab import drive
#drive.mount('/content/gdrive')
#current_directory = 'gdrive/MyDrive/NN-methane/'


property = 'converted_cp'
output_folder = os.path.join(current_directory, "results_experimental-based")
os.makedirs(output_folder, exist_ok=True)
output_file = "NIST_Experimental-based.txt"
header = ['Temp. K', 'Press. MPa', 'Cv-J/mol*K', 'DOI']

# List of molecules to process (See 'Molecules Dictionary' below). Separate them by comma (,)
molecules_to_process = [1]  

# ===================================================
#                MOLECULES DICTIONARY
# ===================================================

switch = {
    # Alkanes
    1: {'ID': 'C74828', 'name': 'Methane'},
    2: {'ID': 'C74840', 'name': 'Ethane'},
    3: {'ID': 'C74986', 'name': 'Propane'},
    4: {'ID': 'C106978', 'name': 'Butane'},
    5: {'ID': 'C75285', 'name': 'Isobutane'},
    6: {'ID': 'C109660', 'name': 'Pentane'},
    7: {'ID': 'C78784', 'name': '2-Methylbutane'},
    8: {'ID': 'C110543', 'name': 'Hexane'},
    9: {'ID': 'C107835', 'name': '2-Methylpentane'},
    10: {'ID': 'C110827', 'name': 'Cyclohexane'},
    11: {'ID': 'C142825', 'name': 'Heptane'},
    12: {'ID': 'C111659', 'name': 'Octane'},
    13: {'ID': 'C111842', 'name': 'Nonane'},
    14: {'ID': 'C124185', 'name': 'Decane'},
    15: {'ID': 'C112403', 'name': 'Dodecane'},

    # Alkenes
    16: {'ID': 'C74851', 'name': 'Ethylene'},
    17: {'ID': 'C115071', 'name': 'Propene'},

    # Alkynes
    18: {'ID': 'C74997', 'name': 'Propyne'},

    # CO2
    19: {'ID': 'C124389', 'name': 'Carbon dioxide'},

    # N2
    20: {'ID': 'C7727379', 'name': 'Nitrogen'},

    # F2
    21: {'ID': 'C7782414', 'name': 'Fluorine'},

    # Noble Gases
    22: {'ID': 'C7440597', 'name': 'Helium'},
    23: {'ID': 'C7440019', 'name': 'Neon'},
    24: {'ID': 'C7440371', 'name': 'Argon'},
    25: {'ID': 'C7439909', 'name': 'Krypton'},
    26: {'ID': 'C7440633', 'name': 'Xenon'},

    # Others
    27: {'ID': 'C7732185', 'name': 'Water'},
    28: {'ID': 'C1333740', 'name': 'Hydrogen'},
    29: {'ID': 'B5000001', 'name': 'Parahydrogen'},
    30: {'ID': 'B5000007', 'name': 'Orthohydrogen'},
    31: {'ID': 'C7782390', 'name': 'Deuterium'},
    32: {'ID': 'C7782447', 'name': 'Oxygen'},
    33: {'ID': 'C630080', 'name': 'Carbon monoxide'},
    34: {'ID': 'C10024972', 'name': 'Dinitrogen monoxide'},
    35: {'ID': 'C7789200', 'name': 'Deuterium oxide'},
    36: {'ID': 'C67561', 'name': 'Methanol'},
    37: {'ID': 'C75194', 'name': 'Cyclopropane'},
    38: {'ID': 'C463821', 'name': '2,2-Dimethylpropene'},
    39: {'ID': 'C7664417', 'name': 'Ammonia'},
    40: {'ID': 'C7783542', 'name': 'Nitrogen trifluoride'},
    41: {'ID': 'C75694', 'name': 'Trichlorofluoromethane (R11)'},
    42: {'ID': 'C75718', 'name': 'Dichlorodifluoromethane (R12)'},
    43: {'ID': 'C75729', 'name': 'Chlorotrifluoromethane (R13)'},
    44: {'ID': 'C75730', 'name': 'Tetrafluoromethane (R14)'},
    45: {'ID': 'C75434', 'name': 'Dichlorofluoromethane (R21)'},
    46: {'ID': 'C75456', 'name': 'Chlorodifluoromethane (R22)'},
    47: {'ID': 'C75467', 'name': 'Trifluoromethane (R23)'},
    48: {'ID': 'C75105', 'name': 'Difluoroethane (R32)'},
    49: {'ID': 'C593533', 'name': 'Fluoromethane (R41)'},
    50: {'ID': 'C76131', 'name': '1,1,2-Trichloro-1,2,2-tetrafluoroethane (R113)'},
    51: {'ID': 'C76142', 'name': '1,2-Dichloro-1,1,2,2-tetrafluoroethane (R114)'},
    52: {'ID': 'C76153', 'name': 'Chloropentafluoroethane (R115)'},
    53: {'ID': 'C76164', 'name': 'Hexafluoroethane (R116)'},
    54: {'ID': 'C306832', 'name': '2,2-dichloro-1,1,1-trifluoroethane (R123)'},
    55: {'ID': 'C2837890', 'name': '1-chloro-1,2,2,2-tetrafluoroethane (R124)'},
    56: {'ID': 'C354336', 'name': 'Pentafluoroethane (R125)'},
    57: {'ID': 'C811972', 'name': '1,1,1,2-tetrafluoroethane (R134a)'},
    58: {'ID': 'C1717006', 'name': '1,1-Dichloro-1-fluoroethane (R141b)'},
    59: {'ID': 'C75683', 'name': '1-Chloro-1,2-difluoroethane (R142b)'},
    60: {'ID': 'C420462', 'name': '1,1,1-trifluoroethane (R143a)'},
    61: {'ID': 'C75376', 'name': '1,1-difluoroethane (R152a)'},
    62: {'ID': 'C76197', 'name': 'Octafluoropropane (R218)'},
    63: {'ID': 'C431890', 'name': '1,1,1,2,3,3,3-Heptafluoropropane (R227ea)'},
    64: {'ID': 'C431630', 'name': '1,1,1,2,3,3-Hexafluoropropane (R236ea)'},
    65: {'ID': 'C690391', 'name': '1,1,1,3,3,3-Hexafluoropropane (R236fa)'},
    66: {'ID': 'C679867', 'name': '1,1,2,2,3-Pentafluoropropane (R245ca)'},
    67: {'ID': 'C460731', 'name': '1,1,1,3,3-Pentafluoropropane (R245fa)'},
    68: {'ID': 'C115253', 'name': 'Octafluorocyclobutane (RC318)'},
    69: {'ID': 'C71432', 'name': 'Benzene'},
    70: {'ID': 'C108883', 'name': 'Toluene'},
    71: {'ID': 'C355259', 'name': 'Decafluorobutane'},
    72: {'ID': 'C678262', 'name': 'Dodecafluoropentane'},
    73: {'ID': 'C7446095', 'name': 'Sulfur dioxide'},
    74: {'ID': 'C7783064', 'name': 'Hydrogen sulfide'},
    75: {'ID': 'C2551624', 'name': 'Sulfur hexafluoride'},
    76: {'ID': 'C463581', 'name': 'Carbonyl sulfide'}
}



# ===================================================
#                    Operations
# ===================================================

# ********************** DATA ACQUISITION **********************
#                   *****  Experimental  *****
Exp_files = pd.read_csv(f'{property}.csv', names=header, header=0)
X_Exp = Exp_files[[header[0], header[1]]].values.tolist()


# Loop through each molecule
for n in molecules_to_process:
    molecule_name = switch[n]['name']
    case_data = switch[n]

    # Flag to check if file has header
    file_has_header = os.path.exists(os.path.join(output_folder, output_file))

    # Loop through each (T_exp, P_exp) in X_Exp
    for index, (T_exp, P_exp) in enumerate(X_Exp):
        try:
            # Construct the URL
            url = f'https://webbook.nist.gov/cgi/fluid.cgi?Action=Data&Wide=on&ID={case_data["ID"]}&Type=IsoBar&Digits=6&P={P_exp}&THigh={T_exp}&TLow={T_exp}&TInc=0.01&RefState=DEF&TUnit=K&PUnit=MPa&DUnit=kg%2Fm3&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm'
            
            # Fetch the data from the URL
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            
            # Save the data to a text file in append mode
            output_path = os.path.join(output_folder, output_file)
            with open(output_path, 'a') as f:
                if not file_has_header:
                    f.write(response.text)
                    file_has_header = True
                else:
                    # Only append data without header (assuming header check is done)
                    lines = response.text.splitlines()
                    if len(lines) > 1:
                        f.write('\n'.join(lines[1:]) + '\n')
        
        except requests.RequestException as e:
            print(f"Failed to fetch data for T_exp={T_exp}, P_exp={P_exp}: {e}")