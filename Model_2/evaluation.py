import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

def evaluate_classification(y_true, y_pred):
    """
    Evaluate the basic classification performance.
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    
    Returns:
    dict: Performance metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_onset_offset_detection(model, X_test, ground_truth_signals, time_tolerance=100, fs=2000):
    """
    Evaluate the onset and offset detection performance.
    
    Parameters:
    model: Trained model
    X_test: Test features
    ground_truth_signals: List of ground truth signals
    time_tolerance: Time tolerance in ms
    fs: Sampling frequency
    
    Returns:
    dict: Performance metrics
    """
    # Time tolerance in samples
    tolerance_samples = int(time_tolerance / 1000 * fs)
    
    all_metrics = {
        'onset': {'precision': [], 'recall': [], 'f1': [], 'mae': []},
        'offset': {'precision': [], 'recall': [], 'f1': [], 'mae': []}
    }
    
    # Predict all at once
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate window size from input dimension
    feature_dim = 3  # LE, RMS, CWT
    window_size = X_test.shape[1] // feature_dim
    
    # Process each signal separately
    start_idx = 0
    for gt_signal in ground_truth_signals:
        # Number of windows for this signal
        n_windows = len(gt_signal) - window_size + 1
        
        # Get predictions for this signal
        signal_pred = y_pred[start_idx:start_idx + n_windows]
        
        # Post-processing: Remove short activations (< 60 samples)
        signal_pred = post_process_predictions(signal_pred)
        
        # Find onset/offset events
        gt_onset_indices, gt_offset_indices = find_events(gt_signal)
        pred_onset_indices, pred_offset_indices = find_events(signal_pred)
        
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
        
        # Update start index for next signal
        start_idx += n_windows
    
    # Calculate average metrics
    avg_metrics = {
        'onset': {
            'precision': np.mean(all_metrics['onset']['precision']),
            'recall': np.mean(all_metrics['onset']['recall']),
            'f1': np.mean(all_metrics['onset']['f1']),
            'mae': np.mean([x for x in all_metrics['onset']['mae'] if x != float('inf')])
        },
        'offset': {
            'precision': np.mean(all_metrics['offset']['precision']),
            'recall': np.mean(all_metrics['offset']['recall']),
            'f1': np.mean(all_metrics['offset']['f1']),
            'mae': np.mean([x for x in all_metrics['offset']['mae'] if x != float('inf')])
        }
    }
    
    return avg_metrics

def post_process_predictions(signal_pred, min_duration=60):
    """
    Apply post-processing to remove short activations.
    
    Parameters:
    signal_pred: Predicted signal
    min_duration: Minimum duration of activation in samples
    
    Returns:
    array: Post-processed predictions
    """
    processed_pred = signal_pred.copy()
    
    for i in range(len(processed_pred) - min_duration):
        # If a sequence of 1s is shorter than min_duration, set to 0
        if processed_pred[i] == 0 and processed_pred[i+1] == 1:
            # Start of potential activation
            activation_start = i+1
            j = i+1
            while j < len(processed_pred) and processed_pred[j] == 1:
                j += 1
            activation_end = j
            
            if activation_end - activation_start < min_duration:
                processed_pred[activation_start:activation_end] = 0
                
        # If a sequence of 0s is shorter than min_duration, set to 1
        if processed_pred[i] == 1 and processed_pred[i+1] == 0:
            # Start of potential silent period
            silent_start = i+1
            j = i+1
            while j < len(processed_pred) and processed_pred[j] == 0:
                j += 1
            silent_end = j
            
            if silent_end - silent_start < min_duration:
                processed_pred[silent_start:silent_end] = 1
    
    return processed_pred

def find_events(signal):
    """
    Find onset and offset events in a signal.
    
    Parameters:
    signal: Binary signal (0 = silent, 1 = active)
    
    Returns:
    tuple: (onset_indices, offset_indices)
    """
    onset_indices = []
    offset_indices = []
    
    for i in range(1, len(signal)):
        if signal[i-1] == 0 and signal[i] == 1:
            onset_indices.append(i)
        elif signal[i-1] == 1 and signal[i] == 0:
            offset_indices.append(i)
    
    return onset_indices, offset_indices

def evaluate_events(pred_indices, gt_indices, tolerance, fs):
    """
    Evaluate event detection performance.
    
    Parameters:
    pred_indices: Predicted event indices
    gt_indices: Ground truth event indices
    tolerance: Tolerance in samples
    fs: Sampling frequency
    
    Returns:
    dict: Performance metrics
    """
    tp = 0
    fp = 0
    mae = []
    
    for pred_idx in pred_indices:
        # Find closest ground truth event
        closest_gt = None
        min_distance = float('inf')
        
        for gt_idx in gt_indices:
            distance = abs(gt_idx - pred_idx)
            if distance < min_distance:
                min_distance = distance
                closest_gt = gt_idx
        
        if closest_gt is not None and min_distance <= tolerance:
            tp += 1
            mae.append(min_distance)
        else:
            fp += 1
    
    fn = len(gt_indices) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Mean absolute error in ms
    mae_ms = np.mean(mae) * 1000 / fs if mae else float('inf')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae_ms
    }

def save_evaluation_results(results, output_dir="evaluation"):
    """
    Save evaluation results to a text file.
    
    Parameters:
    results: Evaluation results
    output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {results['classification']['accuracy']:.4f}\n")
        f.write(f"Test Precision: {results['classification']['precision']:.4f}\n")
        f.write(f"Test Recall: {results['classification']['recall']:.4f}\n")
        f.write(f"Test F1-Score: {results['classification']['f1']:.4f}\n\n")
        
        f.write("Onset Detection Metrics:\n")
        f.write(f"Precision: {results['onset']['precision']:.4f}\n")
        f.write(f"Recall: {results['onset']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['onset']['f1']:.4f}\n")
        f.write(f"MAE: {results['onset']['mae']:.2f} ms\n\n")
        
        f.write("Offset Detection Metrics:\n")
        f.write(f"Precision: {results['offset']['precision']:.4f}\n")
        f.write(f"Recall: {results['offset']['recall']:.4f}\n")
        f.write(f"F1-Score: {results['offset']['f1']:.4f}\n")
        f.write(f"MAE: {results['offset']['mae']:.2f} ms\n")