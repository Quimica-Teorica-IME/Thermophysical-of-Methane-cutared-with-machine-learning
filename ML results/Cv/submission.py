import os

# Get the current directory (root)
root_dir = os.getcwd()

# Enter the 'codes' directory inside the root directory
codes_dir = os.path.join(root_dir, 'codes')

# Get all the .py files in the 'codes' directory
py_files = [f for f in os.listdir(codes_dir) if f.endswith('.py')]

# Create the 'submissions' directory in the root directory
submissions_dir = os.path.join(root_dir, 'submissions')
os.makedirs(submissions_dir, exist_ok=True)

# Iterate over each .py file and create a .sub file for it in the 'submissions' directory
for py_file in py_files:
    model_name = os.path.splitext(py_file)[0]
    sub_file_path = os.path.join(submissions_dir, f'{model_name}.sub')
    
    # Create the content for the .sub file
    sub_file_content = f"""#!/bin/bash
##
## Copyright (C) 2009-2021 VersatusHPC, Inc.
##
## partition = queue
#SBATCH --partition=normal
##
## nodes = number of nodes
#SBATCH --nodes=1
#SBATCH --nodelist=n01
##
## ntasks-per-node = number of cores per node
#SBATCH --ntasks-per-node=256
##
## time = execution time
#SBATCH --time=720:00:00
##
## Job name
#SBATCH --job-name=Cv_{model_name}
##

cd /home/mmaximo/ML/Thermophysical/Cv/codes

# Activate Conda and the MachineLearning-env environment
source ~/.bashrc
conda init
conda activate MachineLearning-env

python {model_name}.py > {model_name}.log 2>&1
"""
    
    # Write the content to the .sub file ensuring Linux line endings and UTF-8 encoding
    with open(sub_file_path, 'w', newline='\n', encoding='utf-8') as sub_file:
        sub_file.write(sub_file_content.lstrip())  # Ensure no leading whitespaces before the #! line

print(f'Successfully created submission scripts for {len(py_files)} models.')
