import os
import re
import csv

def find_log_file(directory):
    for file in os.listdir(directory):
        if file.endswith(".log"):
            return file
    return None


# ============== Results ==============
def extract_metrics_from_lines(lines, start_index):
    metrics = {}
    keys = ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2']
    for i, key in enumerate(keys):
        line = lines[start_index + i + 1].strip()
        if f'{key}:' in line:
            metrics[key] = line.split(f'{key}:')[1].strip()
        else:
            metrics[key] = 'N/A'
    return metrics

def format_metrics(metrics):
    formatted = {
        'MAPE': f"{float(metrics['MAPE'].replace('%', '').strip()):.1f}%" if metrics['MAPE'] != 'N/A' else 'N/A',
        'MAE': f"{float(metrics['MAE']):.3f}" if metrics['MAE'] != 'N/A' else 'N/A',
        'MSE': f"{float(metrics['MSE']):.3f}" if metrics['MSE'] != 'N/A' else 'N/A',
        'RMSE': f"{float(metrics['RMSE']):.3f}" if metrics['RMSE'] != 'N/A' else 'N/A',
        'R2': f"{float(metrics['R2']):.3f}" if metrics['R2'] != 'N/A' else 'N/A'
    }
    return formatted

def write_metrics_to_csv_global(csv_writer, log_file_name, directory_name, phrases, lines):
    states = ['liquid', 'vapor', 'supercritical']
    csv_writer.writerow([f"Directory: {directory_name}"])
    for state in states:
        csv_writer.writerow([f"{state.capitalize()} Phase - {log_file_name}"])
        csv_writer.writerow(['Comparison', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2'])
        for phrase in phrases:
            if state in phrase:
                for i, line in enumerate(lines):
                    if phrase.lower().strip() in line.lower().strip():
                        metrics = extract_metrics_from_lines(lines, i)
                        formatted_metrics = format_metrics(metrics)
                        metric_values = [formatted_metrics.get(name, '') for name in ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2']]
                        comparison = phrase.replace(f"{state} - ", "")
                        csv_writer.writerow([comparison] + metric_values)
        csv_writer.writerow([])  # extra line between phases
    csv_writer.writerow([])  # second empty line between directories


# ============== Cross Validation Metrics ==============
def extract_crossval_blocks(lines):
    """Extract cross-validation metrics per phase."""
    blocks = []
    current_block = []
    inside_block = False

    for line in lines:
        if '*** Cross-Validation Metrics ***' in line:
            inside_block = True
            current_block = []
        elif inside_block:
            current_block.append(line.strip())
            if '>>>' in line:
                blocks.append(current_block)
                inside_block = False
    return blocks

def parse_crossval_block(block):
    """Parse a cross-validation block into structured metrics."""
    text = ' '.join(block)
    text = text.replace("RÂ²", "R²")

    match_phase = re.search(r'(Liquid|Vapor|Supercritical) Phase - RMSE \(mean\):', text, re.IGNORECASE)
    phase = match_phase.group(1).capitalize() if match_phase else "Unknown"

    def extract_metric(name):
        mean_match = re.search(rf'{name} \(mean\): ([\-\d\.Ee]+)', text)
        std_match = re.search(rf'{name} \(std\): ([\-\d\.Ee]+)', text)
        if mean_match and std_match:
            mean = round(float(mean_match.group(1)), 3)
            std = round(float(std_match.group(1)), 3)
            return mean, std
        return None, None

    return {
        'phase': phase,
        'RMSE': extract_metric('RMSE'),
        'MAE': extract_metric('MAE'),
        'R²': extract_metric('R²'),
        'MSE': extract_metric('MSE')
    }    


# ============== Best hyperparameters results ==============
def extract_hyperparameter_lines(lines):
    """Extract all lines containing best hyperparameters."""
    return [line.strip() for line in lines if line.startswith("Best Hyperparameters for")]


# ============== MAIN ==============
def main():
    current_directory = os.getcwd()
    output_csv_path = os.path.join(current_directory, "results.csv")
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as results_file:
        writer = csv.writer(results_file)
        
        for subdir in os.listdir(current_directory):
            subdir_path = os.path.join(current_directory, subdir)
            if not os.path.isdir(subdir_path) or subdir in ['Experimental', 'Lazypredict']:
                continue
            
            log_file = find_log_file(subdir_path)
            if not log_file:
                print(f"[WARNING] No .log file found in {subdir}")
                continue
            
            log_file_path = os.path.join(subdir_path, log_file)
            with open(log_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            phrases = [f"{state} - Exp x Model" for state in ['liquid', 'vapor', 'supercritical']] + \
                      [f"{state} - Exp x NIST" for state in ['liquid', 'vapor', 'supercritical']] + \
                      [f"{state} - NIST x Model" for state in ['liquid', 'vapor', 'supercritical']]
            
            write_metrics_to_csv_global(writer, os.path.splitext(log_file)[0], subdir, phrases, lines)

    print(f"\n✅ Result saved to {output_csv_path}")

    # --- CROSS VALIDATION METRICS ---
    all_crossval = {}

    for subdir in os.listdir(current_directory):
        subdir_path = os.path.join(current_directory, subdir)
        if not os.path.isdir(subdir_path) or subdir in ['Experimental', 'Lazypredict']:
            continue

        log_file = find_log_file(subdir_path)
        if not log_file:
            continue

        with open(os.path.join(subdir_path, log_file), 'r') as f:
            lines = f.readlines()

        blocks = extract_crossval_blocks(lines)
        parsed_blocks = [parse_crossval_block(block) for block in blocks if block]
        if parsed_blocks:
            all_crossval[subdir] = parsed_blocks

    crossval_file = os.path.join(current_directory, "cross_val_metrics.txt")
    with open(crossval_file, 'w', encoding='utf-8') as f:
        for directory, blocks in all_crossval.items():
            f.write(f"Directory: {directory}\n\n")
            for block in blocks:
                f.write(f"---- {block['phase']} Phase ----\n")
                for metric in ['RMSE', 'MAE', 'R²', 'MSE']:
                    mean, std = block[metric]
                    if mean is not None:
                        f.write(f"{metric} (mean): {mean:.3f} (std): {std:.3f}\n")
                    else:
                        f.write(f'Erro aqui {metric}')
                f.write("\n")

    print("✅ Cross-validation metrics saved to cross_val_metrics.txt")

    # --- HYPERPARAMETERS ---
    all_hyperparams = {}

    for subdir in os.listdir(current_directory):
        subdir_path = os.path.join(current_directory, subdir)
        if not os.path.isdir(subdir_path) or subdir in ['Experimental', 'Lazypredict']:
            continue

        log_file = find_log_file(subdir_path)
        if not log_file:
            continue

        with open(os.path.join(subdir_path, log_file), 'r') as f:
            lines = f.readlines()

        hyper_lines = extract_hyperparameter_lines(lines)
        if hyper_lines:
            all_hyperparams[subdir] = hyper_lines

    hyper_file = os.path.join(current_directory, "hyperparameters.txt")
    with open(hyper_file, 'w') as f:
        for directory, lines in all_hyperparams.items():
            f.write(f"========== {directory} ==========\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n\n")


    print("✅ Hyperparameters saved to hyperparameters.txt")


if __name__ == "__main__":
    main()

