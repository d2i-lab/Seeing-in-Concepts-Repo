import subprocess
import pickle
import os

maskclip_python = '/home/jxu680/miniconda3/envs/maskclip/bin/python3'

def get_text_features(text_list):
    # Create a temporary file to store the output
    temp_output_file = 'temp_clip_features.pkl'
    
    # Prepare the command
    command = [
        maskclip_python,
        'clip_encode.py',
        '--word_list'
    ] + text_list + [
        '--output_path',
        temp_output_file
    ]
    
    # Run the subprocess
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running clip_encode.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

    # Load the pickle file
    try:
        with open(temp_output_file, 'rb') as f:
            feature_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading the output file: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)

    return feature_dict