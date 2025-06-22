import argparse
from train import train_demann_model, evaluate_demann_model
import tensorflow as tf

def main():
    """
    Main function to train and evaluate the DEMANN model.
    """
    parser = argparse.ArgumentParser(description='DEMANN: Detector of Muscular Activity by Neural Networks')
    
    parser.add_argument('--dataset_dir', type=str, default='emg_dataset',
                        help='Directory containing the dataset (default: emg_dataset)')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save models and evaluation results (default: output)')
    
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to use (for debugging, default: use all)')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='both',
                        help='Mode: train, evaluate, or both (default: both)')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pre-trained model for evaluation (required if mode is evaluate)')
    
    args = parser.parse_args()
    
    # Print GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("No GPU found, using CPU.")
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*50)
        print("Training DEMANN model...")
        print("="*50)
        
        model = train_demann_model(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_files=args.max_files
        )
        
        if args.mode == 'both':
            print("\n" + "="*50)
            print("Evaluating trained model...")
            print("="*50)
            
            evaluate_demann_model(
                model=model,
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                max_files=args.max_files
            )
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            parser.error("--model_path is required when mode is 'evaluate'")
        
        print("\n" + "="*50)
        print(f"Loading model from {args.model_path}...")
        print("="*50)
        
        model = tf.keras.models.load_model(args.model_path)
        
        print("\n" + "="*50)
        print("Evaluating model...")
        print("="*50)
        
        evaluate_demann_model(
            model=model,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_files=args.max_files
        )

if __name__ == "__main__":
    main()


