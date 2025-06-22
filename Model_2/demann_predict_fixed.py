import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

# Import DEMANN modules - make sure these are available in your path
from model import create_demann_model
from evaluation import post_process_predictions, find_events, evaluate_events

class SignalNormalize:
    """Transform to normalize signals"""
    def __call__(self, signal):
        # Z-score normalization (zero mean, unit variance)
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)  # Add small epsilon to prevent division by zero

class EMGDataset(Dataset):
    """Dataset for EMG signals with onset/offset labels"""
    
    def __init__(self, data_items, transform=None):
        """
        Args:
            data_items: List of dictionaries with npz_path and json_path
            transform: Optional transform to apply to the data
        """
        self.data_items = data_items
        self.transform = transform
        
        # Default signal length
        self.signal_length = 435  # Standard epoch length
        
        print(f"Dataset created with {len(data_items)} items")
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Get file paths
        npz_path = item['npz_path']
        json_path = item['json_path']
        
        # Extract filename for plotting
        filename = os.path.basename(npz_path).split('_emg.npz')[0]
        
        # Try to load the signal data
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Access signal data following the working approach
            signal_dict = data["signal"].item()  # Convert to dictionary
            signal = signal_dict["signal"]
            
            # Make sure signal is a numpy array of float32
            signal = np.asarray(signal, dtype=np.float32)
            
            # If signal is not the right shape, resize it
            if len(signal) != self.signal_length:
                temp_signal = np.zeros(self.signal_length, dtype=np.float32)
                copy_length = min(len(signal), self.signal_length)
                temp_signal[:copy_length] = signal[:copy_length]
                signal = temp_signal
                
        except Exception as e:
            print(f"Error loading signal from {npz_path}: {e}")
            # If we can't load the signal, use zeros
            signal = np.zeros(self.signal_length, dtype=np.float32)
        
        # Create binary mask (0 = no activity, 1 = activity)
        mask = np.zeros(self.signal_length, dtype=np.float32)
        
        # Load label data if available
        if item.get('has_valid_label', False):
            try:
                with open(json_path, 'r') as f:
                    label_data = json.load(f)
                
                # Check for onset/offset directly in the label_data
                onset = label_data.get('onset')
                offset = label_data.get('offset')
                
                if onset is not None and offset is not None:
                    # Make sure onset/offset are within valid range
                    if 0 <= onset < self.signal_length and 0 <= offset < self.signal_length:
                        mask[onset:offset+1] = 1.0
            except Exception as e:
                print(f"Error loading label from {json_path}: {e}")
                # If we can't load the labels, keep zeros
                pass
        
        # Apply any transformations
        if self.transform:
            signal = self.transform(signal)
        
        return signal, mask, filename

def analyze_burst_labels(json_path):
    """
    Analyze the all_burst_labels.json file to find valid/invalid labels
    
    Args:
        json_path: Path to the all_burst_labels.json file
        
    Returns:
        valid_data: List of items with valid labels
        invalid_data: List of items with invalid labels
    """
    with open(json_path, 'r') as f:
        all_data = json.load(f)
    
    valid_data = []
    invalid_data = []
    
    for item in all_data:
        if item.get('has_valid_label', False):
            valid_data.append(item)
        else:
            invalid_data.append(item)
    
    print(f"Total items: {len(all_data)}")
    print(f"Items with valid labels: {len(valid_data)}")
    print(f"Items without valid labels: {len(invalid_data)}")
    
    return valid_data, invalid_data

def extract_demann_features(signals, fs=1000):
    """
    Extract the features needed for DEMANN: LE, RMS, and CWT.
    
    Args:
        signals: EMG signal data, shape (n_samples, time_points)
        fs: Sampling frequency in Hz (default: 1000)
        
    Returns:
        all_features: Combined features for DEMANN, shape (n_samples, n_windows, feature_dim)
    """
    from scipy import signal as signal_processing
    import pywt
    
    n_samples = len(signals)
    time_points = signals[0].shape[0]
    all_features = []
    
    # Butterworth filter setup with corrected frequency normalization
    nyquist = fs / 2
    
    # Ensure filter frequencies are in the valid range (0 < Wn < 1)
    low_freq = min(10/nyquist, 0.99)  # Ensure it's less than 1
    high_freq = min(500/nyquist, 0.99)  # Ensure it's less than 1
    
    # Make sure the frequencies are above 0
    low_freq = max(low_freq, 0.001)
    high_freq = max(high_freq, 0.001)
    
    # If frequencies are too close, adjust them
    if abs(high_freq - low_freq) < 0.001:
        high_freq = min(low_freq + 0.1, 0.99)
    
    for i in tqdm(range(n_samples), desc="Extracting features"):
        emg_signal = signals[i]  # (time_points,)
        
        try:
            # Apply band-pass filter (10-500 Hz)
            b, a = signal_processing.butter(2, [low_freq, high_freq], btype='bandpass')
            filtered_emg = signal_processing.filtfilt(b, a, emg_signal)
            
            # Ensure non-negative signal (full-wave rectification)
            filtered_emg = np.abs(filtered_emg)
            
            # 1. Linear Envelope (LE) - Low-pass filter at 5 Hz
            le_freq = min(5/nyquist, 0.99)
            le_freq = max(le_freq, 0.001)  # Ensure it's above 0
            b_le, a_le = signal_processing.butter(2, le_freq, btype='lowpass')
            le = signal_processing.filtfilt(b_le, a_le, filtered_emg)
            
            # 2. Root Mean Square (RMS) with 60-sample sliding window
            window_size = min(60, len(filtered_emg) // 10)  # Ensure window is not too large
            rms = np.zeros_like(filtered_emg)
            
            # Pad signal for edge handling
            padded_emg = np.pad(filtered_emg, (window_size//2, window_size//2), mode='edge')
            
            for j in range(len(filtered_emg)):
                window = padded_emg[j:j+window_size]
                rms[j] = np.sqrt(np.mean(window**2))
            
            # 3. Continuous Wavelet Transform (CWT)
            scales = np.arange(1, 7)  # 6 levels of decomposition as in the paper
            cwt_coeffs, _ = pywt.cwt(filtered_emg, scales, 'morl')
            
            # Compute scalogram (square of absolute CWT coefficients)
            cwt_scalogram = np.abs(cwt_coeffs)**2
            
            # Reduce dimensionality by averaging across scales
            cwt_feature = np.mean(cwt_scalogram, axis=0)
            
        except Exception as e:
            print(f"Error processing signal {i}: {e}")
            # Use zeros if filter fails
            filtered_emg = np.abs(emg_signal)
            le = filtered_emg
            rms = filtered_emg
            cwt_feature = filtered_emg
        
        # Min-max normalization for each feature
        def min_max_normalize(x):
            min_val = np.min(x)
            max_val = np.max(x)
            if max_val - min_val < 1e-8:
                return np.zeros_like(x)
            return (x - min_val) / (max_val - min_val + 1e-10)
        
        le_norm = min_max_normalize(le)
        rms_norm = min_max_normalize(rms)
        cwt_norm = min_max_normalize(cwt_feature)
        
        # Create sliding windows with size=10 for DEMANN
        window_size = 10
        n_windows = time_points - window_size + 1
        sample_features = []
        
        for j in range(n_windows):
            # Create concatenated feature vector for this window
            window_features = np.concatenate([
                le_norm[j:j+window_size],
                rms_norm[j:j+window_size],
                cwt_norm[j:j+window_size]
            ])
            sample_features.append(window_features)
        
        all_features.append(sample_features)
    
    return np.array(all_features)

def train_demann_model_with_emg_dataset(dataset, output_dir="models", fs=1000, batch_size=32):
    """
    Train a DEMANN model with data from EMGDataset
    
    Args:
        dataset: EMGDataset instance
        output_dir: Directory to save model and results
        fs: Sampling frequency in Hz
        batch_size: Batch size for DataLoader
        
    Returns:
        model: Trained DEMANN model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all signals and masks from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_signals = []
    all_masks = []
    all_filenames = []
    
    for signals, masks, filenames in tqdm(dataloader, desc="Loading dataset"):
        all_signals.extend(signals.numpy())
        all_masks.extend(masks.numpy())
        all_filenames.extend(filenames)
    
    # Convert to numpy arrays
    all_signals = np.array(all_signals)
    all_masks = np.array(all_masks)
    
    # Extract DEMANN features
    print("Extracting DEMANN features...")
    features = extract_demann_features(all_signals, fs)
    n_samples, n_windows, feature_dim = features.shape
    
    # Create window labels
    y_windows = np.zeros(n_samples * n_windows)
    window_size = 10
    
    for i in range(n_samples):
        for j in range(n_windows):
            if j + window_size <= len(all_masks[i]):
                # Label the window based on majority vote
                window_labels = all_masks[i, j:j+window_size]
                y_windows[i*n_windows + j] = 1 if np.mean(window_labels) > 0.5 else 0
    
    # Reshape features for training
    X_windows = features.reshape(-1, feature_dim)
    
    # Split into training and validation sets (80% train, 20% validation)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_windows, y_windows, test_size=0.2, random_state=42
    )
    
    # Create and compile the model
    model = create_demann_model(input_dim=feature_dim)
    
    # Define callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'demann_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print("Training DEMANN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=512,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    return model

def predict_with_demann(model, dataset, output_dir="predictions", fs=1000, batch_size=32):
    """
    Make predictions with a trained DEMANN model
    
    Args:
        model: Trained DEMANN model
        dataset: EMGDataset instance
        output_dir: Directory to save predictions
        fs: Sampling frequency in Hz
        batch_size: Batch size for DataLoader
        
    Returns:
        dict: Dictionary with predictions and evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all signals and masks from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_signals = []
    all_masks = []
    all_filenames = []
    
    for signals, masks, filenames in tqdm(dataloader, desc="Loading dataset"):
        all_signals.extend(signals.numpy())
        all_masks.extend(masks.numpy())
        all_filenames.extend(filenames)
    
    # Convert to numpy arrays
    all_signals = np.array(all_signals)
    all_masks = np.array(all_masks)
    
    # Extract DEMANN features
    print("Extracting DEMANN features...")
    features = extract_demann_features(all_signals, fs)
    n_samples, n_windows, feature_dim = features.shape
    
    # Reshape features for prediction
    X_windows = features.reshape(-1, feature_dim)
    
    # Make predictions
    print("Making predictions...")
    y_pred_prob = model.predict(X_windows)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Reshape predictions by sample
    predictions = []
    for i in range(n_samples):
        sample_pred = y_pred[i*n_windows:(i+1)*n_windows]
        
        # Post-process to remove short activations
        processed_pred = post_process_predictions(sample_pred, min_duration=60)
        
        # Resize prediction to match original signal length
        signal_length = all_masks[i].shape[0]
        resized_pred = np.zeros(signal_length, dtype=int)
        copy_length = min(len(processed_pred), signal_length)
        resized_pred[:copy_length] = processed_pred[:copy_length]
        
        predictions.append(resized_pred)
    
    predictions = np.array(predictions)
    
    # Create results directory
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual prediction plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Evaluate metrics and save plots
    all_metrics = {
        'onset': {'precision': [], 'recall': [], 'f1': [], 'mae': []},
        'offset': {'precision': [], 'recall': [], 'f1': [], 'mae': []}
    }
    
    tolerance_samples = int(0.1 * fs)  # 100ms tolerance
    
    print("Evaluating predictions and saving plots...")
    for i in tqdm(range(n_samples)):
        # Save prediction as NPZ file
        np.savez(
            os.path.join(results_dir, f"{all_filenames[i]}_prediction.npz"),
            signal=all_signals[i],
            ground_truth=all_masks[i],
            prediction=predictions[i]
        )
        
        # Find events
        gt_onset_indices, gt_offset_indices = find_events(all_masks[i])
        pred_onset_indices, pred_offset_indices = find_events(predictions[i])
        
        # Evaluate only if there are ground truth labels
        if len(gt_onset_indices) > 0 and len(gt_offset_indices) > 0:
            # Evaluate onset detection
            onset_metrics = evaluate_events(
                pred_onset_indices, gt_onset_indices, tolerance_samples, fs
            )
            
            # Evaluate offset detection
            offset_metrics = evaluate_events(
                pred_offset_indices, gt_offset_indices, tolerance_samples, fs
            )
            
            # Store metrics
            for key in all_metrics['onset']:
                all_metrics['onset'][key].append(onset_metrics[key])
                all_metrics['offset'][key].append(offset_metrics[key])
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        # Calculate time array (in seconds)
        time = np.arange(len(all_signals[i])) / fs
        
        # Plot signal and regions
        plt.plot(time, all_signals[i], 'b-', alpha=0.7, label='EMG Signal')
        
        # Plot ground truth (if available)
        if np.any(all_masks[i] > 0):
            for onset in gt_onset_indices:
                plt.axvline(x=onset/fs, color='g', linestyle='--', alpha=0.7, label='GT Onset' if onset == gt_onset_indices[0] else "")
            for offset in gt_offset_indices:
                plt.axvline(x=offset/fs, color='r', linestyle='--', alpha=0.7, label='GT Offset' if offset == gt_offset_indices[0] else "")
            plt.fill_between(time, 0, 1, where=all_masks[i] > 0, color='g', alpha=0.2, transform=plt.gca().get_xaxis_transform(), label='GT Activity')
        
        # Plot predictions
        for onset in pred_onset_indices:
            plt.axvline(x=onset/fs, color='m', linestyle='-', alpha=0.7, label='Pred Onset' if onset == pred_onset_indices[0] else "")
        for offset in pred_offset_indices:
            plt.axvline(x=offset/fs, color='c', linestyle='-', alpha=0.7, label='Pred Offset' if offset == pred_offset_indices[0] else "")
        plt.fill_between(time, 0, 1, where=predictions[i] > 0, color='b', alpha=0.2, transform=plt.gca().get_xaxis_transform(), label='Pred Activity')
        
        plt.title(f"EMG Onset/Offset Detection - {all_filenames[i]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{all_filenames[i]}_plot.png"))
        plt.close()
    
    # Calculate average metrics
    avg_metrics = {
        'onset': {
            'precision': np.mean([x for x in all_metrics['onset']['precision'] if not np.isnan(x)]),
            'recall': np.mean([x for x in all_metrics['onset']['recall'] if not np.isnan(x)]),
            'f1': np.mean([x for x in all_metrics['onset']['f1'] if not np.isnan(x)]),
            'mae': np.mean([x for x in all_metrics['onset']['mae'] if x != float('inf')])
        },
        'offset': {
            'precision': np.mean([x for x in all_metrics['offset']['precision'] if not np.isnan(x)]),
            'recall': np.mean([x for x in all_metrics['offset']['recall'] if not np.isnan(x)]),
            'f1': np.mean([x for x in all_metrics['offset']['f1'] if not np.isnan(x)]),
            'mae': np.mean([x for x in all_metrics['offset']['mae'] if x != float('inf')])
        }
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("Onset Detection Metrics:\n")
        f.write(f"Precision: {avg_metrics['onset']['precision']:.4f}\n")
        f.write(f"Recall: {avg_metrics['onset']['recall']:.4f}\n")
        f.write(f"F1-Score: {avg_metrics['onset']['f1']:.4f}\n")
        f.write(f"MAE: {avg_metrics['onset']['mae']:.2f} ms\n\n")
        
        f.write("Offset Detection Metrics:\n")
        f.write(f"Precision: {avg_metrics['offset']['precision']:.4f}\n")
        f.write(f"Recall: {avg_metrics['offset']['recall']:.4f}\n")
        f.write(f"F1-Score: {avg_metrics['offset']['f1']:.4f}\n")
        f.write(f"MAE: {avg_metrics['offset']['mae']:.2f} ms\n")
    
    print("\nOnset Detection Metrics:")
    print(f"Precision: {avg_metrics['onset']['precision']:.4f}")
    print(f"Recall: {avg_metrics['onset']['recall']:.4f}")
    print(f"F1-Score: {avg_metrics['onset']['f1']:.4f}")
    print(f"MAE: {avg_metrics['onset']['mae']:.2f} ms")
    
    print("\nOffset Detection Metrics:")
    print(f"Precision: {avg_metrics['offset']['precision']:.4f}")
    print(f"Recall: {avg_metrics['offset']['recall']:.4f}")
    print(f"F1-Score: {avg_metrics['offset']['f1']:.4f}")
    print(f"MAE: {avg_metrics['offset']['mae']:.2f} ms")
    
    return {
        'predictions': predictions,
        'metrics': avg_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='DEMANN for EMG Onset/Offset Detection')
    
    parser.add_argument('--json_labels_path', type=str, required=True,
                        help='Path to the all_burst_labels.json file')
    parser.add_argument('--output_dir', type=str, default='demann_results',
                        help='Directory to save results')
    parser.add_argument('--sampling_freq', type=int, default=1000,
                        help='Sampling frequency of the EMG data in Hz')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'both'], default='both',
                        help='Mode: train, predict, or both')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pre-trained model for prediction (required if mode is predict)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze burst labels
    print(f"Analyzing burst labels from {args.json_labels_path}...")
    valid_data, invalid_data = analyze_burst_labels(args.json_labels_path)
    
    # Set up transformation
    transform = SignalNormalize()
    
    # Create dataset with valid labels
    dataset = EMGDataset(valid_data, transform=transform)
    
    if args.mode in ['train', 'both']:
        # Train DEMANN model
        model = train_demann_model_with_emg_dataset(
            dataset, 
            output_dir=os.path.join(args.output_dir, 'model'),
            fs=args.sampling_freq
        )
        
        # If mode is both, use the trained model for prediction
        if args.mode == 'both':
            predict_with_demann(
                model,
                dataset,
                output_dir=os.path.join(args.output_dir, 'predictions'),
                fs=args.sampling_freq
            )
    
    elif args.mode == 'predict':
        if args.model_path is None:
            model_path = os.path.join(args.output_dir, 'model', 'demann_model.h5')
            if os.path.exists(model_path):
                args.model_path = model_path
            else:
                parser.error("--model_path is required when mode is 'predict'")
        
        # Load the model
        print(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
        
        # Make predictions
        predict_with_demann(
            model,
            dataset,
            output_dir=os.path.join(args.output_dir, 'predictions'),
            fs=args.sampling_freq
        )
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()