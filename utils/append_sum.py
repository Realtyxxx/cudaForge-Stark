
import csv
import os

file_path = '/home/tanyanxi/workspace/fake_stark/run/20251124_151725_19_ReLU_baseline_doubao_doubao-seed-code-preview-251028/usage.csv'

with open(file_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

input_tokens_sum = 0
output_tokens_sum = 0
total_tokens_sum = 0

for row in rows:
    if not row: continue
    # Assuming the last 3 columns are at indices -3, -2, -1
    # Based on previous file header: timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens
    # Indices are 3, 4, 5
    try:
        input_tokens_sum += int(row[3])
        output_tokens_sum += int(row[4])
        total_tokens_sum += int(row[5])
    except (ValueError, IndexError):
        continue

print(f"Sums: {input_tokens_sum}, {output_tokens_sum}, {total_tokens_sum}")

# Append to file
with open(file_path, 'a') as f:
    writer = csv.writer(f)
    # Create a row with empty values for the first few columns and sums for the last 3
    new_row = ["Total", "", "", input_tokens_sum, output_tokens_sum, total_tokens_sum]
    writer.writerow(new_row)
