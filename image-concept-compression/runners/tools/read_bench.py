import json
import csv
import sys
from pathlib import Path

def parse_json_to_csv(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract metadata
    metadata = data['metadata']

    # Prepare CSV headers
    headers = [
        'dataset', 'index_setting', 'k_coarse', 'm', 'nbits', 'nprobes', 'timestamp',
        'n_segments', 'our_method_mean', 'our_method_std', 'pq_mean', 'pq_std'
    ]

    # Prepare rows for CSV
    rows = []
    for key in data:
        if key.startswith('n_segments_'):
            n_segments = key.split('_')[-1]
            row = [
                metadata['dataset'],
                metadata['index_setting'],
                metadata['k_coarse'],
                metadata['m'],
                metadata['nbits'],
                metadata['nprobes'],
                metadata['timestamp'],
                n_segments,
                data[key]['our_method']['mean'],
                data[key]['our_method']['std'],
                data[key]['pq']['mean'],
                data[key]['pq']['std']
            ]
            rows.append(row)

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_json_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    parse_json_to_csv(input_file, output_file)