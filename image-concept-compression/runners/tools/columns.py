import json
import re
import argparse
import os

def format_output(data, file_name):
    segment_keys = sorted([key for key in data.keys() if key.startswith("n_segments_") and not key.endswith("_raw")],
                          key=lambda x: int(re.search(r'\d+', x).group()))

    print(segment_keys)
    
    print(f"\nFile: {file_name}")
    print("\nAP Statistics:")
    print_stats(data, segment_keys, "ap")
    
    print("\nCounter Statistics:")
    print_stats(data, segment_keys, "counters")

def print_stats(data, segment_keys, metric_type):
    our_means = ["Our Means:"]
    our_stds = ["Our STDs:"]
    pq_means = ["PQ Means:"]
    pq_stds = ["PQ STDs:"]
    ivf_means = ["IVF Means:"]
    ivf_stds = ["IVF STDs:"]

    found_ivf = False
    
    for key in segment_keys:
        if metric_type == "ap":
            our_method = data[key]["our_ap"]
            pq = data[key]["pq_ap"]
            if "ivf_ap" in data[key]:
                ivf = data[key]["ivf_ap"]
            else:
                ivf = None
        else:  # counters
            our_method = data[key]["our_counters"]
            pq = data[key]["pq_counters"]
            if "ivf_counters" in data[key]:
                ivf = data[key]["ivf_counters"]
            else:
                ivf = None
        
        our_means.append(f"{our_method['mean']:.3f}")
        our_stds.append(f"{our_method['std']:.3f}")
        pq_means.append(f"{pq['mean']:.3f}")
        pq_stds.append(f"{pq['std']:.3f}")
        if ivf:
            found_ivf = True
            ivf_means.append(f"{ivf['mean']:.3f}")
            ivf_stds.append(f"{ivf['std']:.3f}")
    
    if found_ivf:
        print(*ivf_means)
        print(*ivf_stds)
        print('--')

    print(*our_means)
    print(*our_stds)
    print('--')
    print(*pq_means)
    print(*pq_stds)

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    format_output(data, os.path.basename(file_path))

def main():
    parser = argparse.ArgumentParser(description="Process JSON files in a directory and output formatted results.")
    parser.add_argument("--dir", '-d', required=True, help="Path to the directory containing JSON files")
    
    args = parser.parse_args()

    # Process all JSON files in the directory
    for file_name in os.listdir(args.dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(args.dir, file_name)
            process_file(file_path)

if __name__ == "__main__":
    main()
