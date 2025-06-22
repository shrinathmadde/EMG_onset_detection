import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def prepare_windows_from_features(le, rms, cwt_feature, ground_truth, window_size=10):
    """
    Prepare the input windows from pre-processed features.
    
    Parameters:
    le (array): Linear envelope feature
    rms (array): RMS feature
    cwt_feature (array): CWT feature
    ground_truth (array): Ground truth labels
    window_size (int): Size of sliding window
    
    Returns:
    tuple: (X_windows, y_windows)
    """
    # Min-max normalization for each feature
    def min_max_normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    
    le_norm = min_max_normalize(le)
    rms_norm = min_max_normalize(rms)
    cwt_norm = min_max_normalize(cwt_feature)
    
    # Create concatenated feature vector
    # The paper mentions they used a concatenation of LE + RMS + CWT features
    feature_vector = np.column_stack([
        le_norm.reshape(-1, 1), 
        rms_norm.reshape(-1, 1),
        # If cwt is 2D, we need to extract a 1D feature from it (paper doesn't specify exactly how)
        # For simplicity, we're using the provided 1D cwt_feature
        cwt_norm.reshape(-1, 1)
    ])
    
    n_samples = len(feature_vector)
    
    # Create sliding windows
    X_windows = []
    y_windows = []
    
    for i in range(n_samples - window_size + 1):
        window = feature_vector[i:i+window_size].flatten()
        
        # Label the window according to the most frequent ground truth value
        window_gt = ground_truth[i:i+window_size]
        label = 1 if np.sum(window_gt) > window_size/2 else 0
        
        X_windows.append(window)
        y_windows.append(label)
    
    return np.array(X_windows), np.array(y_windows)

def load_dataset(dataset_dir, subset='train', max_files=None):
    """
    Load the dataset from the specified directory.
    
    Parameters:
    dataset_dir (str): Directory containing the dataset
    subset (str): 'train' or 'test'
    max_files (int): Maximum number of files to load (for debugging)
    
    Returns:
    tuple: (X_windows, y_windows, ground_truth_signals)
    """
    subset_dir = os.path.join(dataset_dir, subset)
    metadata_file = os.path.join(dataset_dir, f"{subset}_metadata.csv")
    
    if not os.path.exists(subset_dir) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Dataset directory or metadata file not found: {subset_dir}")
    
    metadata = pd.read_csv(metadata_file)
    
    if max_files is not None:
        metadata = metadata.iloc[:max_files]
    
    all_X_windows = []
    all_y_windows = []
    ground_truth_signals = []
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Loading {subset} data"):
        file_path = os.path.join(subset_dir, row['filename'])
        
        try:
            # Load data
            data = np.load(file_path)
            
            # Extract features and ground truth
            le = data['le']
            rms = data['rms']
            cwt_feature = data['cwt_feature']
            ground_truth = data['ground_truth']
            
            # Save ground truth for later evaluation
            ground_truth_signals.append(ground_truth)
            
            # Prepare windows
            X_windows, y_windows = prepare_windows_from_features(
                le, rms, cwt_feature, ground_truth
            )
            
            all_X_windows.append(X_windows)
            all_y_windows.append(y_windows)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Concatenate all windows
    X_windows = np.vstack(all_X_windows) if all_X_windows else np.array([])
    y_windows = np.concatenate(all_y_windows) if all_y_windows else np.array([])
    
    return X_windows, y_windows, ground_truth_signals


