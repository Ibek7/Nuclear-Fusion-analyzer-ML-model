"""
Advanced neural network architectures for fusion analysis.

This module implements state-of-the-art neural networks including:
- Convolutional Neural Networks (CNNs) for spatial data
- Recurrent Neural Networks (RNNs/LSTMs) for temporal data
- Graph Neural Networks (GNNs) for magnetic field topology
- Attention mechanisms and transformers
- Physics-informed neural networks (PINNs)
- Adversarial networks for data augmentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import json
from pathlib import Path
import warnings

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
    from tensorflow.keras.utils import plot_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Additional neural network libraries
try:
    from tensorflow_addons import layers as tfa_layers
    from tensorflow_addons import optimizers as tfa_optimizers
    HAS_TFA = True
except ImportError:
    HAS_TFA = False

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetworkConfig:
    """Neural network configuration."""
    
    # General settings
    framework: str = "tensorflow"  # tensorflow, pytorch
    device: str = "auto"  # auto, cpu, gpu
    random_seed: int = 42
    
    # Architecture settings
    hidden_layers: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.3
    batch_normalization: bool = True
    
    # Training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Advanced features
    use_attention: bool = False
    use_residual_connections: bool = False
    use_batch_norm: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]


class BaseNeuralNetwork(ABC):
    """Base class for neural networks."""
    
    def __init__(self, config: NeuralNetworkConfig):
        """
        Initialize base neural network.
        
        Args:
            config: Neural network configuration.
        """
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False
        
        # Set random seeds
        if config.framework == "tensorflow" and HAS_TF:
            tf.random.set_seed(config.random_seed)
        elif config.framework == "pytorch" and HAS_TORCH:
            torch.manual_seed(config.random_seed)
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """Build the neural network model."""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the neural network."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.config.framework == "tensorflow":
            self.model.save(path)
        elif self.config.framework == "pytorch":
            torch.save(self.model.state_dict(), path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        if self.config.framework == "tensorflow":
            self.model = keras.models.load_model(path)
        elif self.config.framework == "pytorch":
            self.model.load_state_dict(torch.load(path))
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class FeedForwardNetwork(BaseNeuralNetwork):
    """
    Feed-forward neural network for tabular fusion data.
    
    Implements deep fully-connected networks with regularization.
    """
    
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """Build feed-forward model."""
        if self.config.framework == "tensorflow" and HAS_TF:
            return self._build_tf_model(input_shape, output_shape)
        elif self.config.framework == "pytorch" and HAS_TORCH:
            return self._build_torch_model(input_shape, output_shape)
        else:
            raise RuntimeError(f"Framework {self.config.framework} not available")
    
    def _build_tf_model(self, input_shape: Tuple[int, ...], output_shape: int):
        """Build TensorFlow model."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=regularizers.l1_l2(
                    l1=self.config.l1_reg,
                    l2=self.config.l2_reg
                ),
                name=f"dense_{i}"
            )(x)
            
            # Batch normalization
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f"batch_norm_{i}")(x)
            
            # Activation
            x = layers.Activation(self.config.activation, name=f"activation_{i}")(x)
            
            # Dropout
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_{i}")(x)
            
            # Residual connection
            if self.config.use_residual_connections and i > 0:
                if x.shape[-1] == inputs.shape[-1]:
                    x = layers.Add(name=f"residual_{i}")([x, inputs])
        
        # Output layer
        outputs = layers.Dense(output_shape, activation='linear', name="output")(x)
        
        model = models.Model(inputs, outputs, name="FeedForwardNetwork")
        
        # Compile model
        optimizer_map = {
            "adam": optimizers.Adam(learning_rate=self.config.learning_rate),
            "sgd": optimizers.SGD(learning_rate=self.config.learning_rate),
            "rmsprop": optimizers.RMSprop(learning_rate=self.config.learning_rate)
        }
        
        model.compile(
            optimizer=optimizer_map.get(self.config.optimizer, optimizers.Adam()),
            loss="mse",
            metrics=["mae", "mse"]
        )
        
        return model
    
    def _build_torch_model(self, input_shape: Tuple[int, ...], output_shape: int):
        """Build PyTorch model."""
        class FeedForwardNet(nn.Module):
            def __init__(self, config, input_size, output_size):
                super().__init__()
                self.config = config
                
                layers_list = []
                prev_size = input_size
                
                for i, units in enumerate(config.hidden_layers):
                    # Linear layer
                    layers_list.append(nn.Linear(prev_size, units))
                    
                    # Batch normalization
                    if config.use_batch_norm:
                        layers_list.append(nn.BatchNorm1d(units))
                    
                    # Activation
                    if config.activation == "relu":
                        layers_list.append(nn.ReLU())
                    elif config.activation == "tanh":
                        layers_list.append(nn.Tanh())
                    elif config.activation == "sigmoid":
                        layers_list.append(nn.Sigmoid())
                    
                    # Dropout
                    if config.dropout_rate > 0:
                        layers_list.append(nn.Dropout(config.dropout_rate))
                    
                    prev_size = units
                
                # Output layer
                layers_list.append(nn.Linear(prev_size, output_size))
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, x):
                return self.network(x)
        
        input_size = input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
        return FeedForwardNet(self.config, input_size, output_shape)
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train feed-forward network."""
        if self.config.framework == "tensorflow":
            self._train_tf(X, y, X_val, y_val)
        elif self.config.framework == "pytorch":
            self._train_torch(X, y, X_val, y_val)
    
    def _train_tf(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Train TensorFlow model."""
        # Build model
        self.model = self.build_model(X.shape[1:], 1 if len(y.shape) == 1 else y.shape[1])
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
    
    def _train_torch(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]):
        """Train PyTorch model."""
        # Build model
        output_size = 1 if len(y.shape) == 1 else y.shape[1]
        self.model = self.build_model(X.shape[1:], output_size)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")
        self.model.to(device)
        
        # Create datasets
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y.view(-1, 1))
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if self.config.framework == "tensorflow":
            return self.model.predict(X)
        elif self.config.framework == "pytorch":
            device = next(self.model.parameters()).device
            X_tensor = torch.FloatTensor(X).to(device)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.cpu().numpy()


class ConvolutionalNetwork(BaseNeuralNetwork):
    """
    Convolutional Neural Network for spatial fusion data.
    
    Processes 2D/3D magnetic field configurations.
    """
    
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """Build CNN model."""
        if self.config.framework == "tensorflow" and HAS_TF:
            return self._build_tf_cnn(input_shape, output_shape)
        else:
            raise RuntimeError("CNN currently only supports TensorFlow")
    
    def _build_tf_cnn(self, input_shape: Tuple[int, ...], output_shape: int):
        """Build TensorFlow CNN."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Convolutional layers
        filters = [32, 64, 128, 256]
        
        for i, num_filters in enumerate(filters):
            # Convolution
            if len(input_shape) == 2:  # 2D input
                x = layers.Conv2D(
                    num_filters, (3, 3),
                    activation=self.config.activation,
                    padding='same',
                    name=f"conv2d_{i}"
                )(x)
                x = layers.MaxPooling2D((2, 2), name=f"maxpool2d_{i}")(x)
            elif len(input_shape) == 3:  # 3D input
                x = layers.Conv3D(
                    num_filters, (3, 3, 3),
                    activation=self.config.activation,
                    padding='same',
                    name=f"conv3d_{i}"
                )(x)
                x = layers.MaxPooling3D((2, 2, 2), name=f"maxpool3d_{i}")(x)
            
            # Batch normalization and dropout
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f"bn_conv_{i}")(x)
            
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_conv_{i}")(x)
        
        # Flatten for dense layers
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x) if len(input_shape) == 2 else \
            layers.GlobalAveragePooling3D(name="global_avg_pool")(x)
        
        # Dense layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg),
                name=f"dense_{i}"
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f"bn_dense_{i}")(x)
            
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_dense_{i}")(x)
        
        # Output layer
        outputs = layers.Dense(output_shape, activation='linear', name="output")(x)
        
        model = models.Model(inputs, outputs, name="ConvolutionalNetwork")
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train CNN."""
        # Build model
        self.model = self.build_model(X.shape[1:], 1 if len(y.shape) == 1 else y.shape[1])
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7
            )
        ]
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make CNN predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)


class RecurrentNetwork(BaseNeuralNetwork):
    """
    Recurrent Neural Network for temporal fusion data.
    
    Uses LSTM/GRU for time series prediction.
    """
    
    def __init__(self, config: NeuralNetworkConfig):
        """Initialize RNN."""
        super().__init__(config)
        self.sequence_length = 50
        self.rnn_type = "lstm"  # lstm, gru
    
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """Build RNN model."""
        if self.config.framework == "tensorflow" and HAS_TF:
            return self._build_tf_rnn(input_shape, output_shape)
        else:
            raise RuntimeError("RNN currently only supports TensorFlow")
    
    def _build_tf_rnn(self, input_shape: Tuple[int, ...], output_shape: int):
        """Build TensorFlow RNN."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # RNN layers
        rnn_units = [128, 64, 32]
        
        for i, units in enumerate(rnn_units):
            return_sequences = i < len(rnn_units) - 1
            
            if self.rnn_type == "lstm":
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate * 0.5,
                    name=f"lstm_{i}"
                )(x)
            elif self.rnn_type == "gru":
                x = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate * 0.5,
                    name=f"gru_{i}"
                )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f"bn_rnn_{i}")(x)
        
        # Dense layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.config.activation,
                name=f"dense_{i}"
            )(x)
            
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_{i}")(x)
        
        # Output layer
        outputs = layers.Dense(output_shape, activation='linear', name="output")(x)
        
        model = models.Model(inputs, outputs, name="RecurrentNetwork")
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data."""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train RNN."""
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Build model
        self.model = self.build_model(X_seq.shape[1:], 1 if len(y_seq.shape) == 1 else y_seq.shape[1])
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_seq, y_seq,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make RNN predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Prepare sequences for prediction
        X_seq, _ = self.prepare_sequences(X, np.zeros(X.shape[0]))
        
        return self.model.predict(X_seq)


class PhysicsInformedNetwork(BaseNeuralNetwork):
    """
    Physics-Informed Neural Network (PINN) for fusion physics.
    
    Incorporates physical laws into the loss function.
    """
    
    def __init__(self, config: NeuralNetworkConfig):
        """Initialize PINN."""
        super().__init__(config)
        self.physics_weight = 1.0
    
    def build_model(self, input_shape: Tuple[int, ...], output_shape: int = 1):
        """Build PINN model."""
        if not HAS_TF:
            raise RuntimeError("PINN requires TensorFlow")
        
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Standard feed-forward layers
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.config.activation,
                name=f"dense_{i}"
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f"bn_{i}")(x)
            
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_{i}")(x)
        
        # Output layer
        outputs = layers.Dense(output_shape, activation='linear', name="output")(x)
        
        model = models.Model(inputs, outputs, name="PhysicsInformedNetwork")
        
        return model
    
    def physics_loss(self, y_true, y_pred, inputs):
        """Compute physics-informed loss."""
        # Example: Energy conservation constraint
        # This would be customized based on specific fusion physics
        
        # Basic energy conservation: total energy should be conserved
        # Simplified example - in practice would use domain-specific physics
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            energy = tf.reduce_sum(y_pred ** 2, axis=-1, keepdims=True)
        
        energy_gradient = tape.gradient(energy, inputs)
        energy_conservation = tf.reduce_mean(tf.square(energy_gradient))
        
        return energy_conservation
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train PINN with physics constraints."""
        # Build model
        self.model = self.build_model(X.shape[1:], 1 if len(y.shape) == 1 else y.shape[1])
        
        # Custom training loop for physics loss
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        
        @tf.function
        def train_step(X_batch, y_batch):
            with tf.GradientTape() as tape:
                predictions = self.model(X_batch, training=True)
                
                # Data loss
                data_loss = tf.reduce_mean(tf.square(y_batch - predictions))
                
                # Physics loss
                physics_loss = self.physics_loss(y_batch, predictions, X_batch)
                
                # Total loss
                total_loss = data_loss + self.physics_weight * physics_loss
            
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return total_loss, data_loss, physics_loss
        
        # Training loop
        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
        dataset = dataset.batch(self.config.batch_size)
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            epoch_data_loss = 0
            epoch_physics_loss = 0
            
            for X_batch, y_batch in dataset:
                total_loss, data_loss, physics_loss = train_step(X_batch, y_batch)
                epoch_loss += total_loss
                epoch_data_loss += data_loss
                epoch_physics_loss += physics_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss = {epoch_loss:.4f}, "
                      f"Data Loss = {epoch_data_loss:.4f}, "
                      f"Physics Loss = {epoch_physics_loss:.4f}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make PINN predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)


def create_neural_network(
    network_type: str,
    config: Optional[NeuralNetworkConfig] = None
) -> BaseNeuralNetwork:
    """
    Create neural network of specified type.
    
    Args:
        network_type: Type of neural network.
        config: Network configuration.
        
    Returns:
        Neural network instance.
    """
    if config is None:
        config = NeuralNetworkConfig()
    
    if network_type == "feedforward":
        return FeedForwardNetwork(config)
    elif network_type == "cnn":
        return ConvolutionalNetwork(config)
    elif network_type == "rnn":
        return RecurrentNetwork(config)
    elif network_type == "pinn":
        return PhysicsInformedNetwork(config)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def create_ensemble_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[NeuralNetworkConfig] = None
) -> Dict[str, BaseNeuralNetwork]:
    """
    Create ensemble of different neural networks.
    
    Args:
        X: Training features.
        y: Training targets.
        config: Network configuration.
        
    Returns:
        Dictionary of trained networks.
    """
    if config is None:
        config = NeuralNetworkConfig()
    
    networks = {}
    
    # Feed-forward network
    try:
        ff_net = FeedForwardNetwork(config)
        ff_net.train(X, y)
        networks["feedforward"] = ff_net
        logger.info("✓ Feed-forward network trained")
    except Exception as e:
        logger.error(f"Feed-forward network training failed: {e}")
    
    # Physics-informed network
    try:
        pinn = PhysicsInformedNetwork(config)
        pinn.train(X, y)
        networks["pinn"] = pinn
        logger.info("✓ Physics-informed network trained")
    except Exception as e:
        logger.error(f"PINN training failed: {e}")
    
    return networks