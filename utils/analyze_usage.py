
import csv
import os
import glob
import re
from collections import defaultdict
from pathlib import Path

def parse_run_dir(run_dir_name):
    # Try to find the splitter
    mode = None
    if "_baseline_" in run_dir_name:
        splitter = "_baseline_"
        mode = "baseline"
    elif "_stark_" in run_dir_name:
        splitter = "_stark_"
        mode = "stark"
    else:
        return None, None, None

    parts = run_dir_name.split(splitter)
    prefix = parts[0]
    suffix = parts[1]

    # Check if suffix starts with timestamp
    # e.g. 20251124_071122_...
    # Regex: ^\d{8}_\d{6}_
    suffix = re.sub(r'^\d{8}_\d{6}_', '', suffix)

    # Parse suffix for server and model
    if "_" in suffix:
        server, model = suffix.split("_", 1)
    else:
        server = suffix
        model = "unknown"

    return prefix, server, model

def get_operator_name(file_path, batch_dir_name, prefix):
    path_obj = Path(file_path)
    # Check if file is directly in batch_dir
    if path_obj.parent.name == batch_dir_name:
        # It's in the root of the batch dir
        # Use prefix as operator name, but clean it
        # Remove timestamp from start if present
        clean_prefix = re.sub(r'^\d{8}_\d{6}_', '', prefix)
        # Also remove trailing underscores
        clean_prefix = clean_prefix.strip('_')
        return clean_prefix
    else:
        # It's in a subdirectory
        return path_obj.parent.name

def analyze():
    root_dir = '/home/tanyanxi/workspace/fake_stark/run'
    usage_files = glob.glob(os.path.join(root_dir, '**', 'usage.csv'), recursive=True)
    
    # Data structure: stats[(Model, Operator)] = output_tokens
    stats = defaultdict(int)
    
    print(f"{'Model':<40} | {'Operator':<40} | {'Output Tokens':<15}")
    print("-" * 100)

    for file_path in usage_files:
        path_obj = Path(file_path)
        # Find the directory that is a direct child of 'run'
        parts = path_obj.parts
        try:
            run_index = parts.index('run')
            batch_dir_name = parts[run_index + 1]
        except (ValueError, IndexError):
            continue

        prefix, server, model = parse_run_dir(batch_dir_name)
        if not model:
            continue
            
        operator = get_operator_name(file_path, batch_dir_name, prefix)
        
        # Calculate tokens
        current_tokens = 0
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip Total row if present
                    if row.get('timestamp') == 'Total' or row.get('call_type') == 'sum':
                        continue
                    
                    try:
                        current_tokens += int(row.get('output_tokens', 0))
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        stats[(model, operator)] += current_tokens

    # Sort and print
    for (model, operator), tokens in sorted(stats.items()):
        print(f"{model:<40} | {operator:<40} | {tokens:<15}")

if __name__ == "__main__":
    analyze()
