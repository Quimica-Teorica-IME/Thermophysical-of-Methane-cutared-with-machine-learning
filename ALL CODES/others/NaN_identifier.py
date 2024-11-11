import pandas as pd
import numpy as np

# Name of the CSV file
csv_file = "Joule-Thomson.csv"

# Read the CSV file, assuming the first row contains column headers
df = pd.read_csv(csv_file)

# Select the first three columns
columns_to_check = df.columns[:3]  # Get the names of the first three columns

# Function to check if a value is numeric
def is_numeric(value):
    try:
        float(value)  # Try to convert the value to a float
        return True
    except ValueError:
        return False

# Iterate over the columns to display values and check if they are numeric
for column in columns_to_check:
    column_values = df[column]
    total_values = len(column_values)
    
    print(f"\nColumn: {column}")
    print(f"Total values: {total_values}")
    print(f"Values:\n{column_values.to_list()}")  # Display all values as a list
    
    # Check for non-numeric values
    non_numeric = []
    for i, value in enumerate(column_values):
        if not is_numeric(value):
            non_numeric.append((i + 2, value))  # Save the row index and the non-numeric value (add 2 to compensate for the header row)
    
    if non_numeric:
        print(f"\nNon-numeric values found in column '{column}':")
        for row, value in non_numeric:
            print(f"Non-numeric value '{value}' at row {row}")
    else:
        print("No non-numeric values found in this column.")
