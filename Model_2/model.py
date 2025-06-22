import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

def create_demann_model(input_dim=155):
    """
    Create the DEMANN neural network model as described in the paper.
    It's a hidden single-layer fully-connected neural network.
    
    Parameters:
    input_dim (int): Dimension of the input vector (LE + RMS + CWT)
    
    Returns:
    model: Tensorflow Keras model
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with SGD optimizer as specified in the paper
    optimizer = SGD(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model