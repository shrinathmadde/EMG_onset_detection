import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import pywt
from scipy import signal

def generate_semg_signal(fs=2000, duration=1, sigma=100, alpha=1.5, snr_db=10, plot=False):
    """
    Generate simulated sEMG signal as described in the DEMANN paper.
    
    Parameters:
    fs (int): Sampling frequency in Hz
    duration (float): Duration of the signal in seconds
    sigma (float): Standard deviation of the Gaussian in ms
    alpha (float): Time support multiplier
    snr_db (float): Signal-to-noise ratio in dB
    plot (bool): If True, plot the generated signal
    
    Returns:
    tuple: (generated signal, ground truth, simulation parameters)
    """
    # Convert sigma from ms to seconds
    sigma_s = sigma / 1000
    
    # Total number of samples
    n_samples = int(fs * duration)
    time = np.linspace(0, duration, n_samples)
    
    # Generate background noise (Gaussian with zero mean)
    noise_power = 1
    noise = np.random.normal(0, np.sqrt(noise_power), n_samples)
    
    # Calculate signal power based on SNR
    snr_linear = 10 ** (snr_db / 10)
    signal_power = snr_linear * noise_power
    
    # Time support in seconds
    time_support = 2 * alpha * sigma_s
    
    # Generate the Gaussian distribution for muscle activity
    mu = 0.5  # Middle of the signal
    x = np.linspace(0, 1, n_samples)
    gaussian = np.exp(-0.5 * ((x - mu) / (sigma_s / duration)) ** 2)
    
    # Normalize Gaussian to have the desired power
    gaussian = gaussian / np.sqrt(np.mean(gaussian ** 2)) * np.sqrt(signal_power)
    
    # Generate band-limited stochastic process (80-120 Hz)
    random_signal = np.random.normal(0, 1, n_samples)
    b, a = signal.butter(4, [80/(fs/2), 120/(fs/2)], btype='bandpass')
    filtered_random = signal.filtfilt(b, a, random_signal)
    
    # Normalize filtered random signal to have the desired power
    filtered_random = filtered_random / np.sqrt(np.mean(filtered_random ** 2)) * np.sqrt(signal_power)
    
    # Modulate the filtered random signal with the Gaussian envelope
    emg_activity = filtered_random * gaussian
    
    # Add noise to create the complete signal
    semg = emg_activity + noise
    
    # Create ground truth (1 where muscle is active, 0 where inactive)
    # Muscle is active where Gaussian > 0
    ground_truth = np.zeros(n_samples)
    ground_truth[gaussian > 0.01 * np.max(gaussian)] = 1
    
    # Store simulation parameters
    params = {
        'fs': fs,
        'duration': duration,
        'sigma_ms': sigma,
        'alpha': alpha,
        'snr_db': snr_db,
        'time_support_ms': time_support * 1000
    }
    
    # Find onset and offset points
    onset_offset = find_onsets_offsets(ground_truth)
    
    return semg, ground_truth, params, onset_offset, gaussian

def find_onsets_offsets(ground_truth):
    """
    Find onset and offset indices from binary ground truth.
    
    Args:
        ground_truth: Binary array marking active regions
        
    Returns:
        dict: Onset and offset indices
    """
    changes = np.diff(np.concatenate([[0], ground_truth, [0]]))
    onsets = np.where(changes == 1)[0]
    offsets = np.where(changes == -1)[0] - 1  # -1 to get the last 1 before a change
    
    return {'onsets': onsets, 'offsets': offsets}

def preprocess_semg(semg, fs=2000):
    """
    Preprocess the sEMG signal as described in the DEMANN paper.
    
    Parameters:
    semg (array): Raw sEMG signal
    fs (int): Sampling frequency in Hz
    
    Returns:
    tuple: (linear envelope, RMS, CWT scalogram)
    """
    # Ensure non-negative
    semg = np.abs(semg)
    
    # Band-pass filter (10-500 Hz)
    b, a = signal.butter(2, [10/(fs/2), 500/(fs/2)], btype='bandpass')
    filtered_semg = signal.filtfilt(b, a, semg)
    
    # Linear Envelope (LE) - Low-pass filter at 5 Hz
    b_le, a_le = signal.butter(2, 5/(fs/2), btype='lowpass')
    le = signal.filtfilt(b_le, a_le, filtered_semg)
    
    # Root Mean Square (RMS) with 60-sample sliding window
    window_size = 60
    rms = np.zeros_like(filtered_semg)
    
    # Pad signal for edge handling
    padded_semg = np.pad(filtered_semg, (window_size//2, window_size//2), mode='edge')
    
    for i in range(len(filtered_semg)):
        window = padded_semg[i:i+window_size]
        rms[i] = np.sqrt(np.mean(window**2))
    
    # Continuous Wavelet Transform (CWT) with Morse wavelet
    # Using pywt with 'morl' (Morlet) wavelet as an approximation
    scales = np.arange(1, 7)  # 6 levels of decomposition as specified in the paper
    cwt_coeffs, _ = pywt.cwt(filtered_semg, scales, 'morl')
    
    # Compute scalogram (square of absolute CWT coefficients)
    cwt_scalogram = np.abs(cwt_coeffs)**2
    
    # For visualization, reduce to 1D by taking mean across scales
    cwt_feature = np.mean(cwt_scalogram, axis=0)
    
    return le, rms, cwt_feature, filtered_semg

def generate_visualizations(output_dir="visualizations", n_examples=50):
    """
    Generate visualizations of simulated sEMG signals with different parameters.
    
    Parameters:
    output_dir (str): Directory to save visualizations
    n_examples (int): Number of examples to generate for each parameter combination
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter ranges as per the paper
    sigmas = [50, 100, 150]  # ms
    alphas = [1, 1.5, 2, 2.4]
    snrs = [3, 6, 10, 13, 16, 20, 23, 26, 30]  # dB
    
    # Subdirectories for organized output
    raw_dir = os.path.join(output_dir, "raw_signals")
    processed_dir = os.path.join(output_dir, "processed_signals")
    combined_dir = os.path.join(output_dir, "combined_view")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Generate examples
    print("Generating visualizations...")
    for i in tqdm(range(n_examples), desc="Generating examples"):
        # Randomly select parameters
        sigma = np.random.choice(sigmas)
        alpha = np.random.choice(alphas)
        snr = np.random.choice(snrs)
        
        # Generate signal
        semg, ground_truth, params, onset_offset, gaussian = generate_semg_signal(
            sigma=sigma, alpha=alpha, snr_db=snr
        )
        
        # Preprocess signal
        le, rms, cwt_feature, filtered_semg = preprocess_semg(semg)
        
        # 1. Raw signal with onset/offset visualization
        plt.figure(figsize=(15, 6))
        time = np.linspace(0, params['duration'], len(semg))
        
        # Raw signal
        plt.plot(time, semg, 'b-', label='EMG Signal')
        
        # Gaussian envelope
        plt.plot(time, gaussian, 'g--', alpha=0.7, label='Gaussian Envelope')
        
        # Ground truth regions
        plt.fill_between(time, np.min(semg)-0.5, np.max(semg)+0.5, 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        
        # Onset markers
        for onset in onset_offset['onsets']:
            if onset < len(time):
                plt.axvline(x=time[onset], color='g', linestyle='-', label='_nolegend_')
                plt.text(time[onset], np.max(semg)+0.8, 'Onset', 
                         fontsize=9, color='g', rotation=90)
        
        # Offset markers
        for offset in onset_offset['offsets']:
            if offset < len(time):
                plt.axvline(x=time[offset], color='r', linestyle='-', label='_nolegend_')
                plt.text(time[offset], np.max(semg)+0.8, 'Offset', 
                         fontsize=9, color='r', rotation=90)
        
        plt.title(f'Simulated sEMG Signal (σ={sigma}ms, α={alpha}, SNR={snr}dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(raw_dir, f'semg_raw_sigma{sigma}_alpha{alpha}_snr{snr}_{i}.png'))
        plt.close()
        
        # 2. Processed features visualization
        plt.figure(figsize=(15, 10))
        
        # Raw signal
        plt.subplot(4, 1, 1)
        plt.plot(time, semg, 'b-', label='Raw EMG Signal')
        plt.title(f'Raw sEMG Signal (σ={sigma}ms, α={alpha}, SNR={snr}dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Filtered signal
        plt.subplot(4, 1, 2)
        plt.plot(time, filtered_semg, 'g-', label='Filtered EMG')
        plt.title('Band-pass Filtered EMG Signal (10-500 Hz)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Linear Envelope
        plt.subplot(4, 1, 3)
        plt.plot(time, le, 'r-', label='Linear Envelope')
        plt.title('Linear Envelope (LE)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # RMS
        plt.subplot(4, 1, 4)
        plt.plot(time, rms, 'm-', label='RMS')
        plt.title('Root Mean Square (RMS)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(processed_dir, f'semg_processed_sigma{sigma}_alpha{alpha}_snr{snr}_{i}.png'))
        plt.close()
        
        # 3. Combined view with ground truth and features
        plt.figure(figsize=(15, 12))
        
        # Raw signal with ground truth
        plt.subplot(5, 1, 1)
        plt.plot(time, semg, 'b-', label='EMG Signal')
        plt.fill_between(time, np.min(semg), np.max(semg), 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        for onset in onset_offset['onsets']:
            if onset < len(time):
                plt.axvline(x=time[onset], color='g', linestyle='-', label='_nolegend_')
        for offset in onset_offset['offsets']:
            if offset < len(time):
                plt.axvline(x=time[offset], color='r', linestyle='-', label='_nolegend_')
        plt.title(f'Simulated sEMG Signal with Ground Truth (σ={sigma}ms, α={alpha}, SNR={snr}dB)')
        plt.legend(['EMG Signal', 'Active Region', 'Onset', 'Offset'])
        plt.grid(True, alpha=0.3)
        
        # Gaussian envelope
        plt.subplot(5, 1, 2)
        plt.plot(time, gaussian, 'g-', label='Gaussian Envelope')
        plt.title('Gaussian Envelope (Activity Pattern)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Linear Envelope
        plt.subplot(5, 1, 3)
        plt.plot(time, le, 'r-', label='Linear Envelope')
        plt.fill_between(time, np.min(le), np.max(le), 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        plt.title('Linear Envelope (LE) with Ground Truth')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # RMS
        plt.subplot(5, 1, 4)
        plt.plot(time, rms, 'm-', label='RMS')
        plt.fill_between(time, np.min(rms), np.max(rms), 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        plt.title('Root Mean Square (RMS) with Ground Truth')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # CWT feature
        plt.subplot(5, 1, 5)
        plt.plot(time, cwt_feature, 'c-', label='CWT Feature')
        plt.fill_between(time, np.min(cwt_feature), np.max(cwt_feature), 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        plt.title('CWT Feature with Ground Truth')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f'semg_combined_sigma{sigma}_alpha{alpha}_snr{snr}_{i}.png'))
        plt.close()
    
    print(f"Generated {n_examples} visualizations for each view type.")
    print(f"Results saved to: {output_dir}")

def create_visualization_grid(output_dir="visualization_grid"):
    """
    Create a grid of visualizations showing the impact of different parameters.
    
    Parameters:
    output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter ranges
    sigmas = [50, 100, 150]  # ms
    alphas = [1, 1.5, 2, 2.4]
    snrs = [3, 10, 20, 30]  # dB
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # 1. Impact of SNR (with fixed sigma and alpha)
    plt.figure(figsize=(15, 10))
    sigma = 100
    alpha = 1.5
    
    for i, snr in enumerate(snrs):
        plt.subplot(len(snrs), 1, i+1)
        
        # Generate signal
        semg, ground_truth, params, onset_offset, gaussian = generate_semg_signal(
            sigma=sigma, alpha=alpha, snr_db=snr
        )
        
        time = np.linspace(0, params['duration'], len(semg))
        
        # Plot signal
        plt.plot(time, semg, 'b-', label='EMG Signal')
        plt.plot(time, gaussian, 'g--', alpha=0.7, label='Gaussian Envelope')
        
        # Ground truth regions
        plt.fill_between(time, np.min(semg)-0.5, np.max(semg)+0.5, 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        
        # Onset markers
        for onset in onset_offset['onsets']:
            if onset < len(time):
                plt.axvline(x=time[onset], color='g', linestyle='-', label='_nolegend_')
        
        # Offset markers
        for offset in onset_offset['offsets']:
            if offset < len(time):
                plt.axvline(x=time[offset], color='r', linestyle='-', label='_nolegend_')
        
        plt.title(f'SNR = {snr} dB (σ={sigma}ms, α={alpha})')
        plt.grid(True, alpha=0.3)
        
        # Only add legend to first subplot to save space
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_comparison.png'))
    plt.close()
    
    # 2. Impact of sigma (with fixed SNR and alpha)
    plt.figure(figsize=(15, 10))
    snr = 10
    alpha = 1.5
    
    for i, sigma in enumerate(sigmas):
        plt.subplot(len(sigmas), 1, i+1)
        
        # Generate signal
        semg, ground_truth, params, onset_offset, gaussian = generate_semg_signal(
            sigma=sigma, alpha=alpha, snr_db=snr
        )
        
        time = np.linspace(0, params['duration'], len(semg))
        
        # Plot signal
        plt.plot(time, semg, 'b-', label='EMG Signal')
        plt.plot(time, gaussian, 'g--', alpha=0.7, label='Gaussian Envelope')
        
        # Ground truth regions
        plt.fill_between(time, np.min(semg)-0.5, np.max(semg)+0.5, 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        
        # Onset markers
        for onset in onset_offset['onsets']:
            if onset < len(time):
                plt.axvline(x=time[onset], color='g', linestyle='-', label='_nolegend_')
        
        # Offset markers
        for offset in onset_offset['offsets']:
            if offset < len(time):
                plt.axvline(x=time[offset], color='r', linestyle='-', label='_nolegend_')
        
        plt.title(f'σ = {sigma} ms (SNR={snr}dB, α={alpha})')
        plt.grid(True, alpha=0.3)
        
        # Only add legend to first subplot to save space
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma_comparison.png'))
    plt.close()
    
    # 3. Impact of alpha (with fixed SNR and sigma)
    plt.figure(figsize=(15, 10))
    snr = 10
    sigma = 100
    
    for i, alpha in enumerate(alphas):
        plt.subplot(len(alphas), 1, i+1)
        
        # Generate signal
        semg, ground_truth, params, onset_offset, gaussian = generate_semg_signal(
            sigma=sigma, alpha=alpha, snr_db=snr
        )
        
        time = np.linspace(0, params['duration'], len(semg))
        
        # Plot signal
        plt.plot(time, semg, 'b-', label='EMG Signal')
        plt.plot(time, gaussian, 'g--', alpha=0.7, label='Gaussian Envelope')
        
        # Ground truth regions
        plt.fill_between(time, np.min(semg)-0.5, np.max(semg)+0.5, 
                         where=ground_truth == 1, color='y', alpha=0.2, 
                         label='Active Region')
        
        # Onset markers
        for onset in onset_offset['onsets']:
            if onset < len(time):
                plt.axvline(x=time[onset], color='g', linestyle='-', label='_nolegend_')
        
        # Offset markers
        for offset in onset_offset['offsets']:
            if offset < len(time):
                plt.axvline(x=time[offset], color='r', linestyle='-', label='_nolegend_')
        
        plt.title(f'α = {alpha} (SNR={snr}dB, σ={sigma}ms)')
        plt.grid(True, alpha=0.3)
        
        # Only add legend to first subplot to save space
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_comparison.png'))
    plt.close()
    
    print(f"Generated parameter comparison visualizations in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize simulated EMG data with onset/offset points')
    parser.add_argument('--output_dir', type=str, default='emg_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--n_examples', type=int, default=10,
                        help='Number of examples to generate per parameter combination')
    
    args = parser.parse_args()
    
    # Create the main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate random examples
    generate_visualizations(
        output_dir=os.path.join(args.output_dir, 'examples'),
        n_examples=args.n_examples
    )
    
    # Create parameter comparison visualizations
    create_visualization_grid(
        output_dir=os.path.join(args.output_dir, 'parameter_comparison')
    )

if __name__ == "__main__":
    # For direct use with your specific paths:
    # Uncomment and modify these lines to run without command line arguments
    """
    import sys
    sys.argv = [
        "visualize_simulated_data.py",
        "--output_dir", r"C:\EMG_onset_detection\LOL_project\results\simulated_visualizations",
        "--n_examples", "10"
    ]
    """
    main()