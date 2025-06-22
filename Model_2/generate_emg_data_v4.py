import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_two_spike_emg(fs=2000, duration=0.25, sigma=10, snr_db=25,
                           background_noise_level=0.02, non_negative=True,
                           tight_gt=True, plot=True):
    """
    Generate an sEMG signal with exactly 2 tight spikes and continuous background noise.
    
    Parameters:
    - fs: Sampling frequency
    - duration: Duration of signal in seconds
    - sigma: Spike width in ms
    - snr_db: Signal-to-noise ratio (dB)
    - background_noise_level: Variance of background noise
    - non_negative: Rectify signal if True
    - tight_gt: If True, creates very tight ground truth around spike centers
    - plot: Plot the result
    
    Returns:
    - semg: Simulated EMG signal
    - ground_truth: Binary activation mask
    - params: Simulation metadata
    """
    n_samples = int(fs * duration)
    time = np.linspace(0, duration, n_samples)
    sigma_s = sigma / 1000

    # 1. Background noise always present
    background = np.random.normal(0, np.sqrt(background_noise_level), n_samples)
    drift = 0.01 * np.sin(2 * np.pi * 2 * time)
    background += drift

    # 2. Define exact spike times
    spike_times = [0.06, 0.11]  # fixed early in signal

    # 3. Build envelope with 2 spikes
    envelope = np.zeros(n_samples)
    ground_truth = np.zeros(n_samples)

    for t in spike_times:
        center = int(t * fs)
        x = np.arange(n_samples)
        burst = np.exp(-0.5 * ((x - center) / (sigma_s * fs))**2)
        envelope += burst

        if tight_gt:
            gt_width = int(3 * sigma_s * fs)  # very tight window around center
            start = max(center - gt_width // 2, 0)
            end = min(center + gt_width // 2, n_samples)
            ground_truth[start:end] = 1

    # 4. Band-limited EMG-like random signal
    noise = np.random.normal(0, 1, n_samples)
    b, a = signal.butter(4, [80/(fs/2), 120/(fs/2)], btype='bandpass')
    band_noise = signal.filtfilt(b, a, noise)

    # 5. Scale EMG burst to match SNR
    snr_linear = 10 ** (snr_db / 10)
    power_target = snr_linear * background_noise_level
    band_noise *= np.sqrt(power_target / (np.mean(band_noise**2) + 1e-12))

    # 6. Final EMG = spike modulated + background
    spike_signal = band_noise * envelope
    semg = spike_signal + background

    if non_negative:
        semg = np.abs(semg)
        semg += np.random.normal(0, 1e-5, n_samples)

    params = {
        'fs': fs,
        'duration': duration,
        'sigma_ms': sigma,
        'snr_db': snr_db,
        'background_noise_level': background_noise_level,
        'tight_gt': tight_gt,
        'n_spikes': 2
    }

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time, semg, label='EMG Signal', color='blue')
        plt.fill_between(time, 0, np.max(semg), where=ground_truth > 0,
                         color='orange', alpha=0.3, label='Ground Truth')
        plt.title('Two-Spike sEMG Signal with Tight Ground Truth')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return semg, ground_truth, params



def preprocess_semg(semg, fs=2000, strict_non_flat=True):
    """
    Preprocess the sEMG signal as described in the DEMANN paper.
    
    Parameters:
    semg (array): Raw sEMG signal
    fs (int): Sampling frequency in Hz
    strict_non_flat (bool): If True, ensure no flat lines in any processed signal
    
    Returns:
    tuple: (linear envelope, RMS, CWT scalogram, filtered_semg)
    """
    # Ensure the input has no exactly flat lines (add tiny noise if needed)
    if strict_non_flat:
        # Check for constant segments
        diff = np.diff(semg)
        if np.any(diff == 0):
            # Add very small noise to prevent flat lines
            tiny_noise = np.random.normal(0, 1e-6, len(semg))
            semg = semg + tiny_noise
    
    # Band-pass filter (10-500 Hz)
    b, a = signal.butter(2, [10/(fs/2), 500/(fs/2)], btype='bandpass')
    filtered_semg = signal.filtfilt(b, a, semg)
    
    # Add tiny noise to filtered signal to prevent flat lines
    if strict_non_flat:
        tiny_noise = np.random.normal(0, 1e-6, len(filtered_semg))
        filtered_semg = filtered_semg + tiny_noise
    
    # Linear Envelope (LE) - Low-pass filter at 5 Hz
    # First rectify the signal if it's not already
    abs_filtered = np.abs(filtered_semg)
    b_le, a_le = signal.butter(2, 5/(fs/2), btype='lowpass')
    le = signal.filtfilt(b_le, a_le, abs_filtered)
    
    # Add tiny noise to LE to prevent flat lines
    if strict_non_flat:
        tiny_noise = np.random.normal(0, 1e-6, len(le))
        le = le + tiny_noise
    
    # Root Mean Square (RMS) with 60-sample sliding window
    window_size = 60
    rms = np.zeros_like(filtered_semg)
    
    # Pad signal for edge handling
    padded_semg = np.pad(filtered_semg, (window_size//2, window_size//2), mode='edge')
    
    for i in range(len(filtered_semg)):
        window = padded_semg[i:i+window_size]
        rms[i] = np.sqrt(np.mean(window**2))
    
    # Add tiny noise to RMS to prevent flat lines
    if strict_non_flat:
        tiny_noise = np.random.normal(0, 1e-6, len(rms))
        rms = rms + tiny_noise
    
    # Continuous Wavelet Transform (CWT) with Morse wavelet
    # Using pywt with 'morl' (Morlet) wavelet as an approximation
    scales = np.arange(1, 7)  # 6 levels of decomposition as specified in the paper
    cwt_coeffs, _ = pywt.cwt(filtered_semg, scales, 'morl')
    
    # Compute scalogram (square of absolute CWT coefficients)
    cwt_scalogram = np.abs(cwt_coeffs)**2
    
    # Reduce dimensionality of CWT scalogram by averaging across scales
    cwt_feature = np.mean(cwt_scalogram, axis=0)
    
    # Add tiny noise to CWT feature to prevent flat lines
    if strict_non_flat:
        tiny_noise = np.random.normal(0, 1e-6, len(cwt_feature))
        cwt_feature = cwt_feature + tiny_noise
    
    return le, rms, cwt_feature, filtered_semg

def create_dataset(output_dir="emg_dataset", n_signals_per_config=8, non_negative=True):
    """
    Create a dataset of simulated sEMG signals with varying parameters.
    
    Parameters:
    output_dir (str): Directory to save the dataset
    n_signals_per_config (int): Number of signals to generate for each parameter configuration
    non_negative (bool): If True, ensure signals are non-negative
    
    Returns:
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter ranges
    sigmas = [50, 100, 150]  # ms
    alphas = [1, 1.5, 2, 2.4]
    
    # Background noise levels (0.05 = low, 0.1 = medium, 0.2 = high)
    background_levels = [0.05, 0.1, 0.2]
    
    # Number of activations (1-3)
    activation_counts = [1, 2, 3]
    
    # For training set: SNR from 1 to 30 dB (step = 1)
    train_snrs = np.arange(1, 31)
    
    # For test set: Selected SNR values
    test_snrs = [3, 6, 10, 13, 16, 20, 23, 26, 30]
    
    # Create subdirectories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Metadata to save
    train_metadata = []
    test_metadata = []
    
    # Generate training data
    print("Generating training data...")
    for sigma in tqdm(sigmas, desc="Sigma"):
        for alpha in alphas:
            for snr in train_snrs:
                for bg_level in background_levels:
                    for n_act in activation_counts:
                        for i in range(n_signals_per_config):
                            # Generate signal with guaranteed background noise
                            semg, ground_truth, params = generate_two_spike_emg()
                            
                            # Preprocess signal with strict non-flat enforcement
                            le, rms, cwt_feature, filtered_semg = preprocess_semg(semg, strict_non_flat=True)
                            
                            # Save to file
                            filename = f"train_sigma{sigma}_alpha{alpha:.1f}_snr{snr}_bg{bg_level:.2f}_act{n_act}_id{i}"
                            
                            # Save raw and processed signals
                            np.savez(
                                os.path.join(train_dir, f"{filename}.npz"),
                                raw_semg=semg,
                                filtered_semg=filtered_semg,
                                le=le,
                                rms=rms,
                                cwt_feature=cwt_feature,
                                ground_truth=ground_truth
                            )
                            
                            # Add to metadata
                            train_metadata.append({
                                'filename': f"{filename}.npz",
                                **params,
                                'id': i,
                                'non_negative': non_negative
                            })
    
    # Generate test data
    print("Generating test data...")
    for sigma in tqdm(sigmas, desc="Sigma"):
        for alpha in alphas:
            for snr in test_snrs:
                for bg_level in background_levels:
                    for n_act in activation_counts:
                        for i in range(n_signals_per_config):
                            # Generate signal with guaranteed background noise
                            semg, ground_truth, params = generate_two_spike_emg()
                            
                            # Preprocess signal with strict non-flat enforcement
                            le, rms, cwt_feature, filtered_semg = preprocess_semg(semg, strict_non_flat=True)
                            
                            # Verify there are no flat lines before saving
                            if (np.any(np.diff(semg) == 0) or 
                                np.any(np.diff(filtered_semg) == 0) or
                                np.any(np.diff(le) == 0) or
                                np.any(np.diff(rms) == 0) or
                                np.any(np.diff(cwt_feature) == 0)):
                                # Regenerate if we find flat lines
                                continue
                            
                            # Save to file
                            filename = f"test_sigma{sigma}_alpha{alpha:.1f}_snr{snr}_bg{bg_level:.2f}_act{n_act}_id{i}"
                            
                            # Save raw and processed signals
                            np.savez(
                                os.path.join(test_dir, f"{filename}.npz"),
                                raw_semg=semg,
                                filtered_semg=filtered_semg,
                                le=le,
                                rms=rms,
                                cwt_feature=cwt_feature,
                                ground_truth=ground_truth
                            )
                            
                            # Add to metadata
                            test_metadata.append({
                                'filename': f"{filename}.npz",
                                **params,
                                'id': i,
                                'non_negative': non_negative
                            })
    
    # Save metadata
    pd.DataFrame(train_metadata).to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    pd.DataFrame(test_metadata).to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)
    
    print(f"Dataset generated and saved to {output_dir}")
    print(f"Training signals: {len(train_metadata)}")
    print(f"Testing signals: {len(test_metadata)}")