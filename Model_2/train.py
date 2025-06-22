import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import create_demann_model
from data_processing import load_dataset
from evaluation import evaluate_classification, evaluate_onset_offset_detection, save_evaluation_results

def train_demann_model(dataset_dir, output_dir="models", max_files=None):
    """
    Train the DEMANN model on the dataset.
    
    Parameters:
    dataset_dir (str): Directory containing the dataset
    output_dir (str): Directory to save the trained model
    max_files (int): Maximum number of files to load (for debugging)
    
    Returns:
    model: Trained Tensorflow Keras model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    X_train, y_train, _ = load_dataset(dataset_dir, 'train', max_files)
    
    # Input dimension is the window size multiplied by the feature vector size
    input_dim = X_train.shape[1]
    
    # Create model
    model = create_demann_model(input_dim)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'demann_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=512,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model

def plot_training_history(history, output_dir):
    """
    Plot and save the training history.
    
    Parameters:
    history: Training history
    output_dir: Directory to save the plot
    """
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
    plt.close()

def evaluate_demann_model(model, dataset_dir, output_dir="evaluation", max_files=None):
    """
    Evaluate the DEMANN model on the test set.
    
    Parameters:
    model: Trained Tensorflow Keras model
    dataset_dir (str): Directory containing the dataset
    output_dir (str): Directory to save evaluation results
    max_files (int): Maximum number of files to load (for debugging)
    
    Returns:
    dict: Evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    X_test, y_test, ground_truth_signals = load_dataset(dataset_dir, 'test', max_files)
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Evaluate classification performance
    classification_metrics = evaluate_classification(y_test, y_pred)
    
    print(f"Test Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"Test Precision: {classification_metrics['precision']:.4f}")
    print(f"Test Recall: {classification_metrics['recall']:.4f}")
    print(f"Test F1-Score: {classification_metrics['f1']:.4f}")
    
    # Evaluate onset/offset detection
    onset_offset_metrics = evaluate_onset_offset_detection(model, X_test, ground_truth_signals)
    
    print("\nOnset Detection Metrics:")
    print(f"Precision: {onset_offset_metrics['onset']['precision']:.4f}")
    print(f"Recall: {onset_offset_metrics['onset']['recall']:.4f}")
    print(f"F1-Score: {onset_offset_metrics['onset']['f1']:.4f}")
    print(f"MAE: {onset_offset_metrics['onset']['mae']:.2f} ms")
    
    print("\nOffset Detection Metrics:")
    print(f"Precision: {onset_offset_metrics['offset']['precision']:.4f}")
    print(f"Recall: {onset_offset_metrics['offset']['recall']:.4f}")
    print(f"F1-Score: {onset_offset_metrics['offset']['f1']:.4f}")
    print(f"MAE: {onset_offset_metrics['offset']['mae']:.2f} ms")
    
    # Save evaluation results
    results = {
        'classification': classification_metrics,
        'onset': onset_offset_metrics['onset'],
        'offset': onset_offset_metrics['offset']
    }
    
    save_evaluation_results(results, output_dir)
    
    return results


