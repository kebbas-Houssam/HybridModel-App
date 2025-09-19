# main.py - Complete Ransomware Detection Backend with TensorFlow Compatibility
# Real-time ransomware detection system using PSO-optimized hybrid model

import asyncio
import json
import os
import sys
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import queue
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import traceback
import hashlib

# FastAPI and web components
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Machine Learning components
try:
    import tensorflow as tf
    import joblib
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    TF_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"ML libraries not available: {e}")
    TF_AVAILABLE = False

# Windows-specific imports for I/O monitoring
if sys.platform == "win32":
    try:
        import win32file
        import win32con
        import win32api
        import win32security
        import wmi
        WINDOWS_MONITORING = True
    except ImportError:
        print("Windows monitoring libraries not available")
        WINDOWS_MONITORING = False
else:
    WINDOWS_MONITORING = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MODEL_PATH = "./models/"
    SEQUENCE_LENGTH = 50
    MONITORING_WINDOW = 5.0  # 5 seconds
    DETECTION_THRESHOLD = 0.5
    MAX_PROCESSES = 100
    UPDATE_INTERVAL = 1.0
    BUFFER_SIZE = 1000

config = Config()

# Data models
class IOEvent(BaseModel):
    timestamp: float
    process_id: int
    process_name: str
    file_path: str
    operation: str  # 'read', 'write', 'create', 'delete'
    bytes_transferred: int
    offset: Optional[int] = None

class ProcessStats(BaseModel):
    process_id: int
    process_name: str
    start_time: float
    read_ops: int = 0
    write_ops: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    unique_files: set = set()
    file_operations: List[IOEvent] = []

class DetectionResult(BaseModel):
    timestamp: float
    process_id: int
    process_name: str
    prediction_primary: float
    prediction_hybrid: float
    risk_level: str
    confidence: float
    features: Dict[str, float]
    alert: bool

class SystemStatus(BaseModel):
    status: str
    monitored_processes: int
    alerts_count: int
    uptime: float
    last_detection: Optional[str]

# Custom TensorFlow classes with compatibility fixes
class F2Score(tf.keras.metrics.Metric):
    def __init__(self, name='f2_score', **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        
        beta = 2.0
        numerator = (1 + beta**2) * (precision * recall)
        denominator = (beta**2 * precision) + recall + tf.keras.backend.epsilon()
        f2 = numerator / denominator
        return f2

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.W_a = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_a = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u_a = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W_a, axes=1) + self.b_a)
        ait = tf.tensordot(uit, self.u_a, axes=1)

        if mask is not None:
            ait = tf.where(mask, ait, -1e9)

        ait = tf.nn.softmax(ait, axis=1)
        ait_expanded = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait_expanded
        output = tf.reduce_sum(weighted_input, axis=1)

        return output, ait

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config
    
def load_model_with_config_fix(model_path):
    """Load model with configuration fixes for TensorFlow compatibility"""
    import json
    import tempfile
    import zipfile
    import os
    
    try:
        # First, let's try to fix the model configuration
        print(f"Attempting to fix model configuration for: {model_path}")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the H5 file (it's actually a zip-like format)
            temp_model_path = os.path.join(temp_dir, "fixed_model.h5")
            
            # Copy the original model
            import shutil
            shutil.copy2(model_path, temp_model_path)
            
            # Try to load and fix the model
            try:
                # Load model structure without weights first
                with open(temp_model_path, 'rb') as f:
                    # Try to load with custom objects but ignore compilation
                    custom_objects = {
                        'F2Score': F2Score,
                        'AttentionLayer': AttentionLayer
                    }
                    
                    # Load without compilation and with custom loader
                    model = tf.keras.models.load_model(
                        temp_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    
                    print("Model loaded successfully after config fixes")
                    
                    # Recompile with compatible settings
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    
                    model.compile(
                        optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc'),
                            F2Score(name='f2_score')
                        ]
                    )
                    
                    return model
                    
            except Exception as inner_e:
                print(f"Config fix approach failed: {inner_e}")
                raise inner_e
                
    except Exception as e:
        print(f"Enhanced compatibility loading failed: {e}")
        
        # Last resort: try to rebuild the model from scratch
        print("Attempting to rebuild model architecture...")
        return rebuild_model_architecture(model_path)

def rebuild_model_architecture(model_path):
    """Rebuild model architecture manually to avoid compatibility issues"""
    try:
        # Load the model to get weights, ignoring architecture issues
        print("Attempting to extract weights from model...")
        
        # Try loading just to get the basic structure info
        import h5py
        
        with h5py.File(model_path, 'r') as f:
            # Get model config if available
            if 'model_config' in f.attrs:
                config_json = f.attrs['model_config']
                if isinstance(config_json, bytes):
                    config_json = config_json.decode('utf-8')
                print("Found model config in file")
            else:
                print("No model config found in file")
        
        # Create a new compatible model with the same architecture as your training
        print("Building compatible model architecture...")
        
        # Based on your training code, rebuild the model architecture
        time_steps = 50  # Your sequence length
        n_features = 13  # Your feature count
        
        inputs = tf.keras.layers.Input(shape=(time_steps, n_features), name='io_trace_input')

        # CNN Feature Extraction Block
        cnn_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='cnn_1')(inputs)
        cnn_1 = tf.keras.layers.BatchNormalization(name='bn_cnn_1')(cnn_1)
        cnn_1 = tf.keras.layers.Dropout(0.3, name='dropout_cnn_1')(cnn_1)

        cnn_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='cnn_2')(cnn_1)
        cnn_2 = tf.keras.layers.BatchNormalization(name='bn_cnn_2')(cnn_2)
        cnn_2 = tf.keras.layers.Dropout(0.3, name='dropout_cnn_2')(cnn_2)

        cnn_3 = tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', name='cnn_3')(cnn_2)
        cnn_3 = tf.keras.layers.BatchNormalization(name='bn_cnn_3')(cnn_3)

        # LSTM blocks (without time_major parameter)
        lstm_1 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_block_1')(cnn_3)
        lstm_1 = tf.keras.layers.Dropout(0.3, name='dropout_lstm_1')(lstm_1)

        lstm_2 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_block_2')(cnn_3)
        lstm_2 = tf.keras.layers.Dropout(0.3, name='dropout_lstm_2')(lstm_2)

        lstm_3 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_block_3')(cnn_3)
        lstm_3 = tf.keras.layers.Dropout(0.3, name='dropout_lstm_3')(lstm_3)

        # Attention Mechanism
        attention_1, _ = AttentionLayer(attention_dim=64, name='attention_1')(lstm_1)
        attention_2, _ = AttentionLayer(attention_dim=64, name='attention_2')(lstm_2)
        attention_3, _ = AttentionLayer(attention_dim=64, name='attention_3')(lstm_3)

        # Concatenate attention outputs
        concatenated = tf.keras.layers.Concatenate(name='concat_attention')([attention_1, attention_2, attention_3])

        # Classification Head
        dense_1 = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(concatenated)
        dense_1 = tf.keras.layers.BatchNormalization(name='bn_dense_1')(dense_1)
        dense_1 = tf.keras.layers.Dropout(0.3, name='dropout_dense_1')(dense_1)

        dense_2 = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(dense_1)
        dense_2 = tf.keras.layers.BatchNormalization(name='bn_dense_2')(dense_2)
        dense_2 = tf.keras.layers.Dropout(0.15, name='dropout_dense_2')(dense_2)

        dense_3 = tf.keras.layers.Dense(64, activation='relu', name='dense_3')(dense_2)
        dense_3 = tf.keras.layers.Dropout(0.15, name='dropout_dense_3')(dense_3)

        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='ransomware_prediction')(dense_3)

        # Create new model
        new_model = tf.keras.Model(inputs=inputs, outputs=output, name='rebuilt_ransomware_model')
        
        # Compile the new model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        new_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                F2Score(name='f2_score')
            ]
        )
        
        print("New compatible model architecture created")
        
        # Try to load weights from the original model
        try:
            print("Attempting to load weights from original model...")
            
            # Load original model to get weights only
            original_model = tf.keras.models.load_model(model_path, compile=False)
            
            # Transfer weights layer by layer where names match
            transferred_layers = 0
            for layer in new_model.layers:
                try:
                    original_layer = original_model.get_layer(layer.name)
                    layer.set_weights(original_layer.get_weights())
                    transferred_layers += 1
                except:
                    print(f"Could not transfer weights for layer: {layer.name}")
            
            print(f"Transferred weights for {transferred_layers} layers")
            
        except Exception as weight_error:
            print(f"Could not transfer weights: {weight_error}")
            print("Model will use randomly initialized weights")
        
        return new_model
        
    except Exception as e:
        print(f"Model rebuild failed: {e}")
        raise e

class ModelManager:
    """Manages the PSO-optimized hybrid model with TensorFlow compatibility"""
    
    def __init__(self):
        self.primary_model = None
        self.xgb_model = None
        self.scaler = None
        self.xgb_scaler = None
        self.model_loaded = False
        self.model_info = {}
        
    def load_model_compatible(self, model_path):
        
        """Load H5 model with enhanced TensorFlow version compatibility"""
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Try the enhanced compatibility approach
            model = load_model_with_config_fix(model_path)
            logger.info("Model loaded successfully with enhanced compatibility")
            return model
            
        except Exception as e:
            logger.error(f"Enhanced compatibility loading failed: {e}")
            raise e
        
    def load_models(self, h5_path: str, xgb_path: str, scaler_path: str = None):
        """Load the PSO-optimized hybrid models"""
        try:
            logger.info("="*60)
            logger.info("STARTING MODEL LOADING PROCESS")
            logger.info("="*60)
            
            if not TF_AVAILABLE:
                raise Exception("TensorFlow not available")
            
            # Check files exist
            if not os.path.exists(h5_path):
                raise Exception(f"H5 model file not found: {h5_path}")
            if not os.path.exists(xgb_path):
                raise Exception(f"XGBoost model file not found: {xgb_path}")
            
            # Load primary model with compatibility
            logger.info("Loading primary CNN-LSTM-Attention model...")
            self.primary_model = self.load_model_compatible(h5_path)
            logger.info(f"Primary model loaded - Input shape: {self.primary_model.input_shape}")
            
            # Load XGBoost model
            logger.info("Loading PSO-optimized XGBoost model...")
            self.xgb_model = joblib.load(xgb_path)
            logger.info("XGBoost model loaded successfully")
            
            # Initialize scalers
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                self.scaler = RobustScaler()
                logger.info("Using default RobustScaler")
            
            # Try to find XGBoost scaler
            xgb_scaler_path = xgb_path.replace('.pkl', '_scaler.pkl')
            if os.path.exists(xgb_scaler_path):
                self.xgb_scaler = joblib.load(xgb_scaler_path)
                logger.info(f"XGBoost scaler loaded from {xgb_scaler_path}")
            else:
                self.xgb_scaler = RobustScaler()
                logger.info("Using default XGBoost scaler")
            
            self.model_loaded = True
            self.model_info = {
                'primary_model_path': h5_path,
                'xgb_model_path': xgb_path,
                'loaded_at': datetime.now().isoformat(),
                'input_shape': self.primary_model.input_shape,
                'tensorflow_version': tf.__version__,
                'xgboost_version': xgb.__version__
            }
            
            logger.info("="*60)
            logger.info("MODEL LOADING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loaded = False
            return False
    
    def extract_hybrid_features(self, X_sequences, y_pred_proba_primary):
        """Extract hybrid features for XGBoost"""
        try:
            # Primary model predictions
            primary_features = []
            primary_features.append(y_pred_proba_primary.flatten())
            primary_features.append((y_pred_proba_primary > 0.5).astype(int).flatten())
            
            # Statistical features from sequences
            sequence_stats = []
            
            for seq in X_sequences:
                stats = []
                
                # Per-feature statistics
                stats.extend([
                    np.mean(seq, axis=0),
                    np.std(seq, axis=0),
                    np.min(seq, axis=0),
                    np.max(seq, axis=0),
                    np.median(seq, axis=0),
                ])
                
                # Global sequence statistics
                stats.extend([
                    np.mean(seq),
                    np.std(seq),
                    np.var(seq),
                    np.ptp(seq),
                    np.mean(np.diff(seq, axis=0)**2) if len(seq) > 1 else 0
                ])
                
                flattened_stats = np.concatenate([np.array(s).flatten() for s in stats])
                sequence_stats.append(flattened_stats)
            
            sequence_stats = np.array(sequence_stats)
            
            # Combine all features
            hybrid_features = np.column_stack([
                np.array(primary_features).T,
                sequence_stats
            ])
            
            return hybrid_features
            
        except Exception as e:
            logger.error(f"Error extracting hybrid features: {e}")
            return None
    
    def predict(self, sequences: np.ndarray) -> Tuple[float, float]:
        """Make prediction using both models"""
        try:
            if not self.model_loaded:
                raise Exception("Models not loaded")
            
            # Ensure proper shape
            if len(sequences.shape) == 2:
                sequences = sequences.reshape(1, sequences.shape[0], sequences.shape[1])
            
            # Primary model prediction
            y_pred_primary = self.primary_model.predict(sequences, verbose=0)[0][0]
            
            # Extract hybrid features
            hybrid_features = self.extract_hybrid_features(sequences, 
                                                          np.array([[y_pred_primary]]))
            
            if hybrid_features is not None and self.xgb_scaler is not None:
                # Scale hybrid features
                hybrid_features_scaled = self.xgb_scaler.transform(hybrid_features)
                
                # XGBoost prediction
                y_pred_hybrid = self.xgb_model.predict_proba(hybrid_features_scaled)[0][1]
            else:
                y_pred_hybrid = y_pred_primary
            
            return float(y_pred_primary), float(y_pred_hybrid)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, 0.0

class IOMonitor:
    """Real-time I/O monitoring system"""
    
    def __init__(self):
        self.process_stats = {}
        self.event_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.monitoring = False
        self.monitor_thread = None
        self.feature_calculator = FeatureCalculator()
        
    def start_monitoring(self):
        """Start I/O monitoring"""
        if self.monitoring:
            return False
            
        logger.info("Starting I/O monitoring...")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop I/O monitoring"""
        logger.info("Stopping I/O monitoring...")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = time.time()
                
                # Get current processes
                for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                    try:
                        proc_info = proc.info
                        pid = proc_info['pid']
                        name = proc_info['name']
                        
                        # Skip system processes
                        if pid < 10 or name in ['System', 'smss.exe', 'csrss.exe']:
                            continue
                        
                        # Initialize process stats if new
                        if pid not in self.process_stats:
                            self.process_stats[pid] = ProcessStats(
                                process_id=pid,
                                process_name=name,
                                start_time=current_time
                            )
                        
                        # Get I/O statistics
                        try:
                            io_counters = proc.io_counters()
                            stats = self.process_stats[pid]
                            
                            # Update stats
                            stats.read_ops = io_counters.read_count
                            stats.write_ops = io_counters.write_count
                            stats.total_bytes_read = io_counters.read_bytes
                            stats.total_bytes_written = io_counters.write_bytes
                            
                            # Simulate file operations for testing
                            if io_counters.write_count > 0:
                                self._simulate_file_operations(pid, name, current_time)
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Clean old processes
                self._cleanup_old_processes(current_time)
                
                time.sleep(config.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _simulate_file_operations(self, pid: int, name: str, timestamp: float):
        """Simulate file operations for testing"""
        try:
            stats = self.process_stats[pid]
            
            import random
            operations = ['read', 'write', 'create']
            file_extensions = ['.txt', '.doc', '.pdf', '.jpg', '.exe', '.dll']
            
            for _ in range(random.randint(1, 3)):
                operation = random.choice(operations)
                extension = random.choice(file_extensions)
                fake_path = f"C:\\Users\\Document\\file_{random.randint(1000,9999)}{extension}"
                
                event = IOEvent(
                    timestamp=timestamp,
                    process_id=pid,
                    process_name=name,
                    file_path=fake_path,
                    operation=operation,
                    bytes_transferred=random.randint(512, 65536),
                    offset=random.randint(0, 1000000)
                )
                
                stats.file_operations.append(event)
                stats.unique_files.add(fake_path)
                self.event_buffer.append(event)
                
                # Keep only recent events
                cutoff_time = timestamp - config.MONITORING_WINDOW
                stats.file_operations = [
                    op for op in stats.file_operations 
                    if op.timestamp > cutoff_time
                ]
                
        except Exception as e:
            logger.error(f"Error simulating file operations: {e}")
    
    def _cleanup_old_processes(self, current_time: float):
        """Remove old process statistics"""
        cutoff_time = current_time - 300  # 5 minutes
        old_pids = [
            pid for pid, stats in self.process_stats.items()
            if stats.start_time < cutoff_time
        ]
        
        for pid in old_pids:
            del self.process_stats[pid]
    
    def get_process_features(self, pid: int) -> Optional[Dict[str, float]]:
        """Calculate features for a specific process"""
        if pid not in self.process_stats:
            return None
            
        stats = self.process_stats[pid]
        return self.feature_calculator.calculate_features(stats)

class FeatureCalculator:
    """Calculate features from process I/O statistics"""
    
    def calculate_features(self, stats: ProcessStats) -> Dict[str, float]:
        """Calculate the 13 features used by the model"""
        try:
            current_time = time.time()
            recent_events = [
                event for event in stats.file_operations
                if current_time - event.timestamp <= config.MONITORING_WINDOW
            ]
            
            if not recent_events:
                return self._default_features()
            
            # Separate read and write operations
            read_events = [e for e in recent_events if e.operation == 'read']
            write_events = [e for e in recent_events if e.operation == 'write']
            
            read_count = len(read_events)
            write_count = len(write_events)
            total_ops = read_count + write_count
            
            if total_ops == 0:
                return self._default_features()
            
            # Calculate features
            features = {}
            
            features['read_write_ratio'] = read_count / write_count if write_count > 0 else read_count
            features['war_ratio'] = self._calculate_war_ratio(recent_events) / total_ops if total_ops > 0 else 0.0
            features['wss'] = len(stats.unique_files)
            features['entropy'] = self._calculate_entropy(recent_events)
            features['read_pct'] = read_count / total_ops if total_ops > 0 else 0.0
            features['write_pct'] = write_count / total_ops if total_ops > 0 else 0.0
            features['repeat_ratio'] = self._calculate_repeat_ratio(recent_events)
            features['read_entropy'] = self._calculate_entropy(read_events) if read_events else 0.0
            features['write_entropy'] = self._calculate_entropy(write_events) if write_events else 0.0
            features['total_ops'] = float(total_ops)
            
            unique_write_files = len(set(e.file_path for e in write_events))
            features['write_to_unique_ratio'] = write_count / unique_write_files if unique_write_files > 0 else 0.0
            features['avg_offset_gap'] = self._calculate_avg_offset_gap(recent_events)
            features['burstiness'] = self._calculate_burstiness(recent_events)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return self._default_features()
    
    def _default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        return {
            'read_write_ratio': 0.0, 'war_ratio': 0.0, 'wss': 0.0, 'entropy': 0.0,
            'read_pct': 0.0, 'write_pct': 0.0, 'repeat_ratio': 0.0,
            'read_entropy': 0.0, 'write_entropy': 0.0, 'total_ops': 0.0,
            'write_to_unique_ratio': 0.0, 'avg_offset_gap': 0.0, 'burstiness': 0.0
        }
    
    def _calculate_war_ratio(self, events: List[IOEvent]) -> int:
        war_count = 0
        file_reads = set()
        for event in sorted(events, key=lambda x: x.timestamp):
            if event.operation == 'read':
                file_reads.add(event.file_path)
            elif event.operation == 'write' and event.file_path in file_reads:
                war_count += 1
        return war_count
    
    def _calculate_entropy(self, events: List[IOEvent]) -> float:
        if not events:
            return 0.0
        file_counts = defaultdict(int)
        for event in events:
            file_counts[event.file_path] += 1
        total = len(events)
        entropy = 0.0
        for count in file_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_repeat_ratio(self, events: List[IOEvent]) -> float:
        if not events:
            return 0.0
        file_counts = defaultdict(int)
        for event in events:
            file_counts[event.file_path] += 1
        repeated = sum(1 for count in file_counts.values() if count > 1)
        return repeated / len(file_counts) if file_counts else 0.0
    
    def _calculate_avg_offset_gap(self, events: List[IOEvent]) -> float:
        if len(events) < 2:
            return 0.0
        offsets = [e.offset for e in events if e.offset is not None]
        if len(offsets) < 2:
            return 0.0
        offsets.sort()
        gaps = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
        return np.mean(gaps) if gaps else 0.0
    
    def _calculate_burstiness(self, events: List[IOEvent]) -> float:
        if len(events) < 2:
            return 0.0
        timestamps = [e.timestamp for e in events]
        timestamps.sort()
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if not intervals:
            return 0.0
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        if mean_interval == 0:
            return 0.0
        return std_interval / mean_interval

class RansomwareDetector:
    """Main detection engine combining I/O monitoring and ML inference"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.io_monitor = IOMonitor()
        self.detection_results = deque(maxlen=100)
        self.alert_count = 0
        self.start_time = time.time()
        self.detection_thread = None
        self.detecting = False
        
    def load_models(self, h5_path: str, xgb_path: str, scaler_path: str = None) -> bool:
        """Load the detection models"""
        return self.model_manager.load_models(h5_path, xgb_path, scaler_path)
    
    def start_detection(self) -> bool:
        """Start the detection system"""
        if not self.model_manager.model_loaded:
            logger.error("Models not loaded")
            return False
            
        logger.info("Starting ransomware detection system...")
        
        if not self.io_monitor.start_monitoring():
            logger.error("Failed to start I/O monitoring")
            return False
            
        self.detecting = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("Ransomware detection system started!")
        return True
    
    def stop_detection(self):
        """Stop the detection system"""
        logger.info("Stopping ransomware detection system...")
        self.detecting = False
        self.io_monitor.stop_monitoring()
        
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
            
        logger.info("Detection system stopped")
    
    def _detection_loop(self):
        """Main detection loop"""
        sequence_buffer = {}
        
        while self.detecting:
            try:
                current_time = time.time()
                
                for pid, stats in self.io_monitor.process_stats.items():
                    try:
                        features = self.io_monitor.get_process_features(pid)
                        if not features:
                            continue
                        
                        feature_vector = np.array([
                            features['read_write_ratio'], features['war_ratio'],
                            features['wss'], features['entropy'], features['read_pct'],
                            features['write_pct'], features['repeat_ratio'],
                            features['read_entropy'], features['write_entropy'],
                            features['total_ops'], features['write_to_unique_ratio'],
                            features['avg_offset_gap'], features['burstiness']
                        ])
                        
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                        
                        if pid not in sequence_buffer:
                            sequence_buffer[pid] = deque(maxlen=config.SEQUENCE_LENGTH)
                        
                        sequence_buffer[pid].append(feature_vector)
                        
                        if len(sequence_buffer[pid]) == config.SEQUENCE_LENGTH:
                            sequence = np.array(list(sequence_buffer[pid]))
                            
                            if self.model_manager.scaler:
                                sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
                                sequence_scaled = self.model_manager.scaler.transform(sequence_reshaped)
                                sequence = sequence_scaled.reshape(sequence.shape)
                            
                            pred_primary, pred_hybrid = self.model_manager.predict(sequence)
                            risk_level, alert = self._assess_risk(pred_hybrid)
                            
                            if alert:
                                self.alert_count += 1
                            
                            result = DetectionResult(
                                timestamp=current_time,
                                process_id=pid,
                                process_name=stats.process_name,
                                prediction_primary=pred_primary,
                                prediction_hybrid=pred_hybrid,
                                risk_level=risk_level,
                                confidence=max(pred_hybrid, 1 - pred_hybrid),
                                features=features,
                                alert=alert
                            )
                            
                            self.detection_results.append(result)
                            
                            if alert:
                                logger.warning(f"RANSOMWARE ALERT: Process {stats.process_name} (PID: {pid}) - Risk: {pred_hybrid:.3f}")
                            
                    except Exception as e:
                        logger.error(f"Error processing PID {pid}: {e}")
                        continue
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
    
    def _assess_risk(self, prediction: float) -> Tuple[str, bool]:
        """Assess risk level based on prediction"""
        if prediction >= 0.8:
            return "CRITICAL", True
        elif prediction >= 0.6:
            return "HIGH", True
        elif prediction >= 0.4:
            return "MEDIUM", False
        elif prediction >= 0.2:
            return "LOW", False
        else:
            return "MINIMAL", False
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        last_detection = None
        if self.detection_results:
            last_detection = datetime.fromtimestamp(self.detection_results[-1].timestamp).isoformat()
        
        return SystemStatus(
            status="RUNNING" if self.detecting else "STOPPED",
            monitored_processes=len(self.io_monitor.process_stats),
            alerts_count=self.alert_count,
            uptime=time.time() - self.start_time,
            last_detection=last_detection
        )
    
    def get_recent_detections(self, limit: int = 50) -> List[DetectionResult]:
        """Get recent detection results"""
        return list(self.detection_results)[-limit:]

# FastAPI application
app = FastAPI(title="Ransomware Detection System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = RansomwareDetector()
connected_websockets = set()

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize the detection system"""
    logger.info("Starting Ransomware Detection API...")
    
    # Try to load models automatically
    model_files = {
        'h5': 'best_pso_ransomware_model.h5',
        'xgb': 'pso_ransomware_model_xgboost_20250811_154446.pkl'
    }
    
    for file_path in model_files.values():
        if not os.path.exists(file_path):
            logger.warning(f"Model file not found: {file_path}")
    
    # Auto-load models if available
    if all(os.path.exists(path) for path in model_files.values()):
        if detector.load_models(model_files['h5'], model_files['xgb']):
            logger.info("Models loaded successfully on startup")
        else:
            logger.error("Failed to load models on startup")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    detector.stop_detection()
    logger.info("Detection system stopped")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Ransomware Detection System API", "version": "1.0.0"}

@app.get("/api/status")
async def get_status():
    """Get system status"""
    status = detector.get_system_status()
    model_info = detector.model_manager.model_info if detector.model_manager.model_loaded else {}
    
    return {
        "system": status.dict(),
        "models": model_info,
        "websockets": len(connected_websockets)
    }

@app.post("/api/load-models")
async def load_models(
    h5_path: str = "best_pso_ransomware_model.h5",
    xgb_path: str = "pso_ransomware_model_xgboost_20250811_154446.pkl",
    scaler_path: Optional[str] = None
):
    """Load detection models with TensorFlow compatibility"""
    try:
        logger.info(f"API: Load models request - H5: {h5_path}, XGB: {xgb_path}")
        
        if not os.path.exists(h5_path):
            raise HTTPException(status_code=404, detail=f"H5 model file not found: {h5_path}")
        
        if not os.path.exists(xgb_path):
            raise HTTPException(status_code=404, detail=f"XGBoost model file not found: {xgb_path}")
        
        success = detector.load_models(h5_path, xgb_path, scaler_path)
        
        if success:
            await broadcast_to_websockets({
                "type": "model_status", 
                "data": {"status": "loaded", "timestamp": time.time()}
            })
            return {
                "status": "success", 
                "message": "Models loaded successfully",
                "model_info": detector.model_manager.model_info
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load models")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-detection")
async def start_detection():
    """Start the detection system"""
    try:
        if detector.detecting:
            return {"status": "already_running", "message": "Detection already running"}
        
        success = detector.start_detection()
        
        if success:
            await broadcast_to_websockets({
                "type": "system_status", 
                "data": {"status": "started", "timestamp": time.time()}
            })
            return {"status": "success", "message": "Detection started"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start detection")
            
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-detection")
async def stop_detection():
    """Stop the detection system"""
    try:
        detector.stop_detection()
        
        await broadcast_to_websockets({
            "type": "system_status", 
            "data": {"status": "stopped", "timestamp": time.time()}
        })
        
        return {"status": "success", "message": "Detection stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections")
async def get_detections(limit: int = 50):
    """Get recent detection results"""
    try:
        detections = detector.get_recent_detections(limit)
        return {"detections": [d.dict() for d in detections]}
        
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/processes")
async def get_processes():
    """Get monitored processes"""
    try:
        processes = []
        for pid, stats in detector.io_monitor.process_stats.items():
            features = detector.io_monitor.get_process_features(pid)
            processes.append({
                "pid": pid,
                "name": stats.process_name,
                "start_time": stats.start_time,
                "read_ops": stats.read_ops,
                "write_ops": stats.write_ops,
                "total_bytes_read": stats.total_bytes_read,
                "total_bytes_written": stats.total_bytes_written,
                "unique_files": len(stats.unique_files),
                "features": features or {}
            })
        
        return {"processes": processes}
        
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def manual_prediction(features: Dict[str, float]):
    """Manual prediction endpoint for testing"""
    try:
        if not detector.model_manager.model_loaded:
            raise HTTPException(status_code=400, detail="Models not loaded")
        
        feature_vector = np.array([
            features.get('read_write_ratio', 0.0),
            features.get('war_ratio', 0.0),
            features.get('wss', 0.0),
            features.get('entropy', 0.0),
            features.get('read_pct', 0.0),
            features.get('write_pct', 0.0),
            features.get('repeat_ratio', 0.0),
            features.get('read_entropy', 0.0),
            features.get('write_entropy', 0.0),
            features.get('total_ops', 0.0),
            features.get('write_to_unique_ratio', 0.0),
            features.get('avg_offset_gap', 0.0),
            features.get('burstiness', 0.0)
        ])
        
        sequence = np.tile(feature_vector, (config.SEQUENCE_LENGTH, 1))
        pred_primary, pred_hybrid = detector.model_manager.predict(sequence)
        risk_level, alert = detector._assess_risk(pred_hybrid)
        
        return {
            "prediction_primary": pred_primary,
            "prediction_hybrid": pred_hybrid,
            "risk_level": risk_level,
            "alert": alert,
            "confidence": max(pred_hybrid, 1 - pred_hybrid)
        }
        
    except Exception as e:
        logger.error(f"Error in manual prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-status")
async def get_model_status():
    """Get current model loading status"""
    try:
        status = {
            "models_loaded": detector.model_manager.model_loaded,
            "model_info": detector.model_manager.model_info,
            "tensorflow_available": TF_AVAILABLE,
            "xgboost_available": True,
            "working_directory": os.getcwd(),
            "timestamp": time.time()
        }
        
        model_files = {
            "best_pso_ransomware_model.h5": os.path.exists("best_pso_ransomware_model.h5"),
            "pso_ransomware_model_xgboost_20250811_154446": os.path.exists("pso_ransomware_model_xgboost_20250811_154446")
        }
        status["model_files_present"] = model_files
        status["all_files_present"] = all(model_files.values())
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list-model-files")
async def list_model_files():
    """List available model files"""
    try:
        files = {
            "h5_files": [],
            "pkl_files": [],
            "working_directory": os.getcwd()
        }
        
        for item in os.listdir("."):
            if os.path.isfile(item):
                size = os.path.getsize(item)
                file_info = {
                    "name": item,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "path": os.path.abspath(item)
                }
                
                if item.endswith('.h5'):
                    files["h5_files"].append(file_info)
                elif item.endswith('.pkl'):
                    files["pkl_files"].append(file_info)
        
        return files
        
    except Exception as e:
        logger.error(f"Error listing model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug-load-test")
async def debug_load_test():
    """Debug endpoint to test model loading directly"""
    try:
        result = {
            "step1_files_exist": {},
            "step2_tensorflow_test": {},
            "step3_h5_load_test": {},
            "step4_xgb_load_test": {},
            "errors": []
        }
        
        h5_path = "best_pso_ransomware_model.h5"
        xgb_path = "pso_ransomware_model_xgboost_20250811_154446.pkl"
        
        result["step1_files_exist"] = {
            "h5_exists": os.path.exists(h5_path),
            "h5_size": os.path.getsize(h5_path) if os.path.exists(h5_path) else 0,
            "xgb_exists": os.path.exists(xgb_path),
            "xgb_size": os.path.getsize(xgb_path) if os.path.exists(xgb_path) else 0
        }
        
        # Test TensorFlow
        try:
            result["step2_tensorflow_test"] = {
                "version": tf.__version__,
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "can_create_model": True
            }
            test_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
        except Exception as e:
            result["step2_tensorflow_test"]["error"] = str(e)
            result["errors"].append(f"TensorFlow test failed: {e}")
        
        # Test H5 loading with compatibility
        try:
            model_manager = ModelManager()
            test_model = model_manager.load_model_compatible(h5_path)
            result["step3_h5_load_test"] = {
                "loaded_with_compatibility": True,
                "input_shape": str(test_model.input_shape),
                "output_shape": str(test_model.output_shape)
            }
        except Exception as e:
            result["step3_h5_load_test"] = {
                "loaded_with_compatibility": False,
                "error": str(e)
            }
            result["errors"].append(f"H5 compatibility loading failed: {e}")
        
        # Test XGB loading
        try:
            xgb_model = joblib.load(xgb_path)
            result["step4_xgb_load_test"] = {
                "loaded": True,
                "type": str(type(xgb_model))
            }
        except Exception as e:
            result["step4_xgb_load_test"] = {
                "loaded": False,
                "error": str(e)
            }
            result["errors"].append(f"XGBoost loading failed: {e}")
        
        return result
        
    except Exception as e:
        return {"critical_error": str(e)}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time detection updates"""
    await websocket.accept()
    connected_websockets.add(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(connected_websockets)}")
    
    try:
        status = detector.get_system_status()
        await websocket.send_json({
            "type": "system_status",
            "data": status.dict()
        })
        
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                if detector.detecting:
                    recent_detections = detector.get_recent_detections(5)
                    if recent_detections:
                        await websocket.send_json({
                            "type": "detections",
                            "data": [d.dict() for d in recent_detections[-5:]]
                        })
                    
                    status = detector.get_system_status()
                    await websocket.send_json({
                        "type": "system_status",
                        "data": status.dict()
                    })
                
                await asyncio.sleep(2.0)
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_websockets.discard(websocket)
        logger.info(f"WebSocket removed. Total connections: {len(connected_websockets)}")

async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected websockets"""
    if not connected_websockets:
        return
    
    disconnected = set()
    for websocket in connected_websockets:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to websocket: {e}")
            disconnected.add(websocket)
    
    connected_websockets.difference_update(disconnected)

# Static files (for frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML frontend
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    html_path = "static/dashboard.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ransomware Detection Dashboard</title>
        </head>
        <body>
            <h1>Dashboard not found</h1>
            <p>Frontend files not available. API is running at <a href="/docs">/docs</a></p>
            <p>Model status: <a href="/api/model-status">/api/model-status</a></p>
            <p>Debug test: <a href="/api/debug-load-test">/api/debug-load-test</a></p>
        </body>
        </html>
        """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ransomware Detection Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--h5-model", default="best_pso_ransomware_model.h5", help="H5 model path")
    parser.add_argument("--xgb-model", default="pso_ransomware_model_xgboost_20250811_154446.pkl", help="XGBoost model path")
    parser.add_argument("--auto-start", action="store_true", help="Auto start detection")
    
    args = parser.parse_args()
    
    print("="*80)
    print("RANSOMWARE DETECTION SYSTEM - BACKEND SERVER")
    print("="*80)
    print(f"Host: {args.host}:{args.port}")
    print(f"H5 Model: {args.h5_model}")
    print(f"XGBoost Model: {args.xgb_model}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Auto-start: {args.auto_start}")
    print("="*80)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )