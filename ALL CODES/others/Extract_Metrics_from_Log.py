import os
import csv

def find_log_file(directory):
    """Find the first .log file in the given directory."""
    for file in os.listdir(directory):
        if file.endswith(".log"):
            return file
    return None

def extract_metrics_from_lines(lines, start_index):
    """Extract metrics from specific lines in the log file."""
    metrics = {}
    keys = ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2']
    for i, key in enumerate(keys):
        line = lines[start_index + i + 1].strip()
        metrics[key] = line.split(f'{key}:')[1].strip()
    return metrics

def format_metrics(metrics):
    """Format metrics according to specifications."""
    formatted = {
        'MAPE': f"{float(metrics['MAPE'].replace('%', '').strip()):.1f}%",
        'MAE': f"{float(metrics['MAE']):.3f}",
        'MSE': f"{float(metrics['MSE']):.3f}",
        'RMSE': f"{float(metrics['RMSE']):.3f}",
        'R2': f"{float(metrics['R2']):.3f}"
    }
    return formatted

def write_metrics_to_csv(csv_file, log_file_name, phrases, lines):
    """Write the metrics to a CSV file."""
    states = ['liquid', 'vapor', 'supercritical']
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for state in states:
            writer.writerow([f"{state.capitalize()} Phase - {log_file_name}"])
            writer.writerow(['Comparison', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2'])
            for phrase in phrases:
                if state in phrase:
                    for i, line in enumerate(lines):
                        if phrase in line:
                            metrics = extract_metrics_from_lines(lines, i)
                            formatted_metrics = format_metrics(metrics)
                            metric_values = [formatted_metrics.get(name, '') for name in ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2']]
                            comparison = phrase.replace(f"{state} - ", "")
                            writer.writerow([comparison] + metric_values)
            writer.writerow([])

def process_directory(directory):
    """Process each directory to extract metrics and write them to CSV."""
    log_file = find_log_file(directory)
    
    if not log_file:
        print(f"No .log file found in directory: {directory}")
        return
    
    log_file_name = os.path.splitext(log_file)[0]
    log_file_path = os.path.join(directory, log_file)
    print(f"Processing {log_file_name} in directory {directory}")
    
    states = ['liquid', 'vapor', 'supercritical']
    phrases = [f"{state} - Exp x Model" for state in states] + \
              [f"{state} - Exp x NIST" for state in states] + \
              [f"{state} - NIST x Model" for state in states]
    
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    csv_file = os.path.join(directory, f'{log_file_name}_extracted.csv')
    write_metrics_to_csv(csv_file, log_file_name, phrases, lines)
    print(f"Metrics saved in {csv_file}")

def main():
    current_directory = os.getcwd()
    
    for subdir in os.listdir(current_directory):
        subdir_path = os.path.join(current_directory, subdir)
        
        # Skip non-directories and the 'experimental' directory
        if not os.path.isdir(subdir_path) or subdir == 'Experimental' or subdir == "Lazypredict":
            continue
        
        process_directory(subdir_path)

if __name__ == "__main__":
    main()
