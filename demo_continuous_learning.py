#!/usr/bin/env python3
"""
Continuous Learning Pipeline Demo

This demo showcases the continuous learning capabilities including:
- Online/offline feature engineering parity
- Model performance monitoring and drift detection
- Automated retraining triggers
- Ensemble prediction methods
- Model versioning and deployment pipeline
- Incremental learning capabilities

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from services.continuous_learning import (
    ContinuousLearningPipeline,
    FeatureEngineer,
    ModelPerformanceMonitor,
    EnsemblePredictor,
    ModelVersionManager,
    IncrementalLearner,
    FeatureVector,
    ModelMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_flight_data(num_flights: int = 100) -> pd.DataFrame:
    """Create sample flight data for demonstration"""
    np.random.seed(42)
    
    airlines = ['AI', 'UK', '6E', 'SG', 'G8']
    aircraft_types = ['A320', 'B737', 'A321', 'B738', 'ATR72']
    airports = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU']
    
    flights = []
    base_time = datetime(2024, 1, 15, 6, 0, 0)
    
    for i in range(num_flights):
        flight_time = base_time + timedelta(minutes=i * 15)
        
        # Simulate some delays
        scheduled_delay = 0
        if np.random.random() < 0.3:  # 30% chance of delay
            scheduled_delay = np.random.exponential(20)  # Average 20 min delay
        
        actual_time = flight_time + timedelta(minutes=scheduled_delay)
        
        flight = {
            'flight_id': f'{np.random.choice(airlines)}{1000 + i}',
            'flight_no': f'{np.random.choice(airlines)}{1000 + i}',
            'date_local': flight_time.date(),
            'origin': np.random.choice(airports),
            'destination': np.random.choice(airports),
            'aircraft_type': np.random.choice(aircraft_types),
            'std_utc': flight_time.isoformat(),
            'atd_utc': actual_time.isoformat() if scheduled_delay > 0 else None,
            'sta_utc': (flight_time + timedelta(hours=2)).isoformat(),
            'ata_utc': (actual_time + timedelta(hours=2)).isoformat() if scheduled_delay > 0 else None,
            'dep_delay_min': scheduled_delay if scheduled_delay > 0 else 0,
            'arr_delay_min': scheduled_delay if scheduled_delay > 0 else 0
        }
        flights.append(flight)
    
    return pd.DataFrame(flights)

def demo_feature_engineering():
    """Demonstrate feature engineering with online/offline parity"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING DEMO")
    print("="*60)
    
    feature_engineer = FeatureEngineer()
    
    # Create sample data
    flight_data = {
        'flight_id': 'AI2739',
        'std_utc': '2024-01-15 08:30:00'
    }
    
    context_data = {
        'turnaround_time': 45.0,
        'runway_utilization': 0.85,
        'weather_score': 0.2,
        'airline_performance': 0.82,
        'inbound_delay': 12.0,
        'connections': 3.0
    }
    
    # Online feature extraction
    print("1. Online Feature Extraction:")
    online_features = feature_engineer.extract_features_online(flight_data, context_data)
    print(f"   Flight ID: {online_features.flight_id}")
    print(f"   Source: {online_features.source}")
    print(f"   Feature count: {len(online_features.features)}")
    print(f"   Sample features:")
    for name, value in list(online_features.features.items())[:5]:
        print(f"     {name}: {value:.3f}")
    print(f"   Feature hash: {online_features.feature_hash[:8]}...")
    
    # Offline feature extraction
    print("\n2. Offline Feature Extraction:")
    flight_df = pd.DataFrame([flight_data])
    offline_features = feature_engineer.extract_features_offline(flight_df)
    
    print(f"   Processed flights: {len(offline_features)}")
    print(f"   Source: {offline_features[0].source}")
    print(f"   Feature count: {len(offline_features[0].features)}")
    
    # Compare parity
    print("\n3. Online/Offline Parity Check:")
    temporal_features = ['hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour']
    parity_check = True
    
    for feature in temporal_features:
        online_val = online_features.features.get(feature, 0)
        offline_val = offline_features[0].features.get(feature, 0)
        match = abs(online_val - offline_val) < 1e-6
        parity_check = parity_check and match
        print(f"   {feature}: Online={online_val:.3f}, Offline={offline_val:.3f}, Match={match}")
    
    print(f"\n   ✓ Parity Check: {'PASSED' if parity_check else 'FAILED'}")
    
    # Fit transformations
    print("\n4. Fitting Transformations:")
    sample_data = create_sample_flight_data(50)
    all_features = feature_engineer.extract_features_offline(sample_data)
    feature_engineer.fit_transformations(all_features)
    
    print(f"   Fitted scaling parameters for {len(feature_engineer.scaler_params)} features")
    print("   Sample scaling parameters:")
    for name, (mean, std) in list(feature_engineer.scaler_params.items())[:3]:
        print(f"     {name}: mean={mean:.3f}, std={std:.3f}")

def demo_performance_monitoring():
    """Demonstrate model performance monitoring and drift detection"""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMO")
    print("="*60)
    
    monitor = ModelPerformanceMonitor(drift_threshold=0.15)
    
    # Simulate baseline performance
    print("1. Baseline Model Performance:")
    baseline_predictions = np.random.normal(15, 8, 100)  # Mean delay 15 min
    baseline_actuals = baseline_predictions + np.random.normal(0, 3, 100)  # Add noise
    
    baseline_metrics = monitor.evaluate_model_performance(
        'delay_predictor_v1', baseline_predictions.tolist(), baseline_actuals.tolist()
    )
    
    print(f"   Model ID: {baseline_metrics.model_id}")
    print(f"   MAE: {baseline_metrics.mae:.2f} minutes")
    print(f"   Accuracy: {baseline_metrics.accuracy:.3f}")
    print(f"   AUC: {baseline_metrics.auc:.3f}")
    print(f"   Predictions: {baseline_metrics.prediction_count}")
    print(f"   Drift Score: {baseline_metrics.drift_score:.3f}")
    print(f"   Confidence: {baseline_metrics.confidence_score:.3f}")
    
    # Simulate degraded performance
    print("\n2. Degraded Model Performance:")
    degraded_predictions = np.random.normal(20, 12, 100)  # Higher mean, more variance
    degraded_actuals = baseline_actuals  # Same actuals
    
    degraded_metrics = monitor.evaluate_model_performance(
        'delay_predictor_v1', degraded_predictions.tolist(), degraded_actuals.tolist()
    )
    
    print(f"   MAE: {degraded_metrics.mae:.2f} minutes (vs {baseline_metrics.mae:.2f})")
    print(f"   Accuracy: {degraded_metrics.accuracy:.3f} (vs {baseline_metrics.accuracy:.3f})")
    print(f"   Drift Score: {degraded_metrics.drift_score:.3f}")
    
    # Drift detection
    print("\n3. Drift Detection:")
    drift_report = monitor.detect_drift('delay_predictor_v1', degraded_metrics)
    
    print(f"   Drift Detected: {drift_report.drift_detected}")
    print(f"   Drift Score: {drift_report.drift_score:.3f}")
    print(f"   Threshold: {drift_report.threshold:.3f}")
    print(f"   Recommendation: {drift_report.recommendation}")
    
    if drift_report.drift_detected:
        print("   🚨 Model retraining recommended!")
    else:
        print("   ✓ Model performance within acceptable range")

def demo_ensemble_prediction():
    """Demonstrate ensemble prediction methods"""
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION DEMO")
    print("="*60)
    
    ensemble = EnsemblePredictor()
    
    # Create mock models with different prediction patterns
    print("1. Creating Ensemble Models:")
    
    class MockModel:
        def __init__(self, bias, variance):
            self.bias = bias
            self.variance = variance
        
        def predict(self, X):
            return [self.bias + np.random.normal(0, self.variance)]
    
    # Conservative model (low predictions, low variance)
    conservative_model = MockModel(bias=10, variance=2)
    ensemble.add_model('conservative', conservative_model, weight=0.4)
    
    # Aggressive model (high predictions, high variance)
    aggressive_model = MockModel(bias=25, variance=5)
    ensemble.add_model('aggressive', aggressive_model, weight=0.3)
    
    # Balanced model (medium predictions, medium variance)
    balanced_model = MockModel(bias=17, variance=3)
    ensemble.add_model('balanced', balanced_model, weight=0.3)
    
    print(f"   Added {len(ensemble.models)} models to ensemble")
    print("   Model weights:")
    for model_id, weight in ensemble.model_weights.items():
        print(f"     {model_id}: {weight:.1f}")
    
    # Make ensemble predictions
    print("\n2. Ensemble Predictions:")
    
    test_features = FeatureVector(
        flight_id='AI2739',
        timestamp=datetime.now(),
        features={
            'hour_of_day': 8.0,
            'runway_demand_ratio': 0.85,
            'weather_severity_score': 0.2,
            'preceding_delay': 12.0
        },
        feature_hash='test_hash',
        source='test'
    )
    
    # Make multiple predictions to show variability
    predictions = []
    for i in range(5):
        pred = ensemble.predict_ensemble(test_features)
        predictions.append(pred)
        print(f"   Prediction {i+1}:")
        print(f"     Ensemble: {pred.prediction:.1f} minutes")
        print(f"     Confidence: {pred.confidence:.3f}")
        print(f"     Uncertainty: {pred.uncertainty:.3f}")
        print(f"     Components: {', '.join(f'{k}={v:.1f}' for k, v in pred.component_predictions.items())}")
    
    # Show ensemble statistics
    ensemble_preds = [p.prediction for p in predictions]
    print(f"\n3. Ensemble Statistics (5 predictions):")
    print(f"   Mean: {np.mean(ensemble_preds):.1f} minutes")
    print(f"   Std Dev: {np.std(ensemble_preds):.1f} minutes")
    print(f"   Range: {np.min(ensemble_preds):.1f} - {np.max(ensemble_preds):.1f} minutes")
    
    # Update weights based on mock performance
    print("\n4. Updating Model Weights:")
    mock_metrics = {
        'conservative': ModelMetrics('conservative', datetime.now(), mae=8.0, accuracy=0.85, 
                                   auc=0.75, prediction_count=100, drift_score=0.05, confidence_score=0.9),
        'aggressive': ModelMetrics('aggressive', datetime.now(), mae=12.0, accuracy=0.75, 
                                 auc=0.70, prediction_count=100, drift_score=0.08, confidence_score=0.8),
        'balanced': ModelMetrics('balanced', datetime.now(), mae=9.5, accuracy=0.82, 
                               auc=0.78, prediction_count=100, drift_score=0.06, confidence_score=0.85)
    }
    
    print("   Performance metrics:")
    for model_id, metrics in mock_metrics.items():
        print(f"     {model_id}: MAE={metrics.mae:.1f}, Accuracy={metrics.accuracy:.3f}")
    
    ensemble.update_weights(mock_metrics)
    
    print("   Updated weights (based on inverse MAE):")
    for model_id, weight in ensemble.model_weights.items():
        print(f"     {model_id}: {weight:.3f}")

# Create a serializable model class at module level
class MockDelayModel:
    def __init__(self, version):
        self.version = version
        self.feature_importances_ = np.random.random(10)
    
    def predict(self, X):
        return np.random.normal(15 + self.version, 5, len(X))

def demo_model_versioning():
    """Demonstrate model versioning and deployment pipeline"""
    print("\n" + "="*60)
    print("MODEL VERSIONING DEMO")
    print("="*60)
    
    # Create temporary model store
    import tempfile
    temp_dir = tempfile.mkdtemp()
    version_manager = ModelVersionManager(temp_dir)
    
    print(f"1. Model Store Location: {temp_dir}")
    
    # Save multiple model versions
    print("\n2. Saving Model Versions:")
    versions = []
    
    for i in range(3):
        model = MockDelayModel(i)
        metrics = ModelMetrics(
            f'delay_predictor_v{i+1}', datetime.now(), 
            mae=10.0 - i, accuracy=0.75 + i*0.05, auc=0.70 + i*0.03,
            prediction_count=1000, drift_score=0.05, confidence_score=0.85 + i*0.05
        )
        
        feature_importance = {f'feature_{j}': float(model.feature_importances_[j]) for j in range(10)}
        training_hash = f'training_data_hash_v{i+1}'
        
        version_id = version_manager.save_model_version(
            'delay_predictor', model, metrics, feature_importance, training_hash
        )
        versions.append(version_id)
        
        print(f"   Saved version: {version_id}")
        print(f"     MAE: {metrics.mae:.1f}")
        print(f"     Accuracy: {metrics.accuracy:.3f}")
    
    print(f"\n   Total versions saved: {len(version_manager.versions)}")
    
    # Deploy a model
    print("\n3. Model Deployment:")
    deploy_version = versions[1]  # Deploy middle version
    success = version_manager.deploy_model(deploy_version)
    
    print(f"   Deployed version: {deploy_version}")
    print(f"   Deployment success: {success}")
    print(f"   Active model: {version_manager.active_models.get('delay_predictor', 'None')}")
    
    # Load active model
    active_model = version_manager.load_active_model('delay_predictor')
    print(f"   Active model loaded: {active_model is not None}")
    
    # Rollback demonstration
    print("\n4. Model Rollback:")
    rollback_success = version_manager.rollback_model('delay_predictor')
    
    print(f"   Rollback success: {rollback_success}")
    print(f"   New active model: {version_manager.active_models.get('delay_predictor', 'None')}")
    
    # Cleanup demonstration
    print("\n5. Version Cleanup:")
    print(f"   Versions before cleanup: {len(version_manager.versions)}")
    version_manager.cleanup_old_versions(keep_versions=2)
    print(f"   Versions after cleanup: {len(version_manager.versions)}")
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   Cleaned up temporary directory")

# Create a simple model class for incremental learning demo
class SimpleIncrementalModel:
    def __init__(self):
        self.weights = np.random.random(4)
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return np.dot(X, self.weights)
    
    def fit(self, X, y):
        # Simple linear regression update
        if len(X) > 0:
            self.weights = np.linalg.lstsq(X, y, rcond=None)[0]

def demo_incremental_learning():
    """Demonstrate incremental learning capabilities"""
    print("\n" + "="*60)
    print("INCREMENTAL LEARNING DEMO")
    print("="*60)
    
    # Use larger batch size to avoid LightGBM warnings
    learner = IncrementalLearner(batch_size=10, learning_rate=0.05)
    
    print("1. Incremental Learning Setup:")
    print(f"   Batch size: {learner.batch_size}")
    print(f"   Learning rate: {learner.learning_rate}")
    print(f"   Initial buffer size: {len(learner.update_buffer)}")
    
    # Simulate streaming data
    print("\n2. Streaming Training Data:")
    
    for i in range(25):  # More samples for better LightGBM training
        # Create sample feature vector
        features = FeatureVector(
            flight_id=f'AI{2000+i}',
            timestamp=datetime.now() + timedelta(minutes=i*15),
            features={
                'hour_of_day': 8.0 + i * 0.5,
                'runway_demand_ratio': 0.7 + i * 0.02,
                'weather_severity_score': 0.1 + i * 0.01,
                'preceding_delay': i * 2.0
            },
            feature_hash=f'hash_{i}',
            source='streaming'
        )
        
        # Simulate target (delay in minutes)
        target = 10.0 + i * 1.5 + np.random.normal(0, 2)
        
        print(f"   Sample {i+1}: Flight {features.flight_id}, Target delay: {target:.1f} min")
        
        # Add to learner
        learner.add_training_sample(features, target)
        
        print(f"     Buffer size: {len(learner.update_buffer)}")
        
        # Check if model was updated
        if len(learner.update_buffer) == 0 and i > 0:
            print(f"     🔄 Model updated after batch!")
    
    print(f"\n   Final buffer size: {len(learner.update_buffer)}")
    print(f"   Model trained: {learner.model is not None}")
    
    # Test predictions
    if learner.model is not None:
        print("\n3. Making Predictions:")
        
        test_features = [
            FeatureVector(
                flight_id='TEST1',
                timestamp=datetime.now(),
                features={'hour_of_day': 9.0, 'runway_demand_ratio': 0.8, 
                         'weather_severity_score': 0.15, 'preceding_delay': 5.0},
                feature_hash='test1',
                source='test'
            ),
            FeatureVector(
                flight_id='TEST2',
                timestamp=datetime.now(),
                features={'hour_of_day': 14.0, 'runway_demand_ratio': 0.9, 
                         'weather_severity_score': 0.25, 'preceding_delay': 15.0},
                feature_hash='test2',
                source='test'
            )
        ]
        
        for features in test_features:
            prediction = learner.predict(features)
            print(f"   {features.flight_id}: Predicted delay = {prediction:.1f} minutes")

# Create a pipeline model class at module level
class MockPipelineModel:
    def predict(self, X):
        # Simulate higher delays for high utilization
        base_delay = 12.0
        utilization_factor = X[0][1] * 10  # Assume second feature is utilization
        return [base_delay + utilization_factor + np.random.normal(0, 3)]

def demo_full_pipeline():
    """Demonstrate the complete continuous learning pipeline"""
    print("\n" + "="*60)
    print("COMPLETE PIPELINE DEMO")
    print("="*60)
    
    # Initialize pipeline with persistent model store
    config = {
        'retrain_threshold': 0.12,
        'monitoring_interval': 1800,  # 30 minutes
        'auto_deploy': False,
        'model_store_path': 'demo_models',  # Persistent directory
        'batch_size': 50,
        'learning_rate': 0.01
    }
    
    pipeline = ContinuousLearningPipeline(config)
    
    print("1. Pipeline Initialization:")
    print(f"   Retrain threshold: {pipeline.retrain_threshold}")
    print(f"   Monitoring interval: {pipeline.monitoring_interval} seconds")
    print(f"   Auto-deploy: {pipeline.auto_deploy}")
    
    # Simulate online feature extraction
    print("\n2. Online Feature Processing:")
    
    flight_data = {
        'flight_id': 'AI2739',
        'std_utc': '2024-01-15 14:30:00'
    }
    
    context_data = {
        'turnaround_time': 50.0,
        'runway_utilization': 0.92,
        'weather_score': 0.3,
        'airline_performance': 0.78,
        'inbound_delay': 18.0
    }
    
    features = pipeline.update_features_online(flight_data, context_data)
    print(f"   Extracted features for flight: {features.flight_id}")
    print(f"   Feature count: {len(features.features)}")
    print(f"   High runway utilization: {features.features['runway_demand_ratio']:.2f}")
    print(f"   Afternoon peak: {features.features['hour_of_day']:.0f}:30")
    
    # Add mock model to ensemble
    
    print("1. Pipeline Initialization:")
    print(f"   Retrain threshold: {pipeline.retrain_threshold}")
    print(f"   Monitoring interval: {pipeline.monitoring_interval} seconds")
    print(f"   Auto-deploy: {pipeline.auto_deploy}")
    
    # Simulate online feature extraction
    print("\n2. Online Feature Processing:")
    
    flight_data = {
        'flight_id': 'AI2739',
        'std_utc': '2024-01-15 14:30:00'
    }
    
    context_data = {
        'turnaround_time': 50.0,
        'runway_utilization': 0.92,
        'weather_score': 0.3,
        'airline_performance': 0.78,
        'inbound_delay': 18.0
    }
    
    features = pipeline.update_features_online(flight_data, context_data)
    print(f"   Extracted features for flight: {features.flight_id}")
    print(f"   Feature count: {len(features.features)}")
    print(f"   High runway utilization: {features.features['runway_demand_ratio']:.2f}")
    print(f"   Afternoon peak: {features.features['hour_of_day']:.0f}:30")
    
    # Add mock model to ensemble
    
    pipeline.ensemble_predictor.add_model('production_model', MockPipelineModel(), weight=1.0)
    
    # Make ensemble prediction
    print("\n3. Ensemble Prediction:")
    prediction = pipeline.predict_with_ensemble(features)
    
    print(f"   Predicted delay: {prediction.prediction:.1f} minutes")
    print(f"   Confidence: {prediction.confidence:.3f}")
    print(f"   Uncertainty: {prediction.uncertainty:.3f}")
    
    if prediction.prediction > 15:
        print("   ⚠️  High delay risk detected!")
    else:
        print("   ✓ Normal delay risk")
    
    # Simulate model monitoring
    print("\n4. Model Performance Monitoring:")
    
    # Generate sample predictions and actuals
    sample_predictions = [prediction.prediction + np.random.normal(0, 2) for _ in range(20)]
    sample_actuals = [p + np.random.normal(0, 4) for p in sample_predictions]
    
    drift_report = pipeline.monitor_model_drift(sample_predictions, sample_actuals)
    
    print(f"   Drift detected: {drift_report.drift_detected}")
    print(f"   Drift score: {drift_report.drift_score:.3f}")
    print(f"   Recommendation: {drift_report.recommendation}")
    
    # Check retraining trigger
    should_retrain = pipeline.trigger_retraining(pipeline.retrain_threshold)
    print(f"   Retraining needed: {should_retrain}")
    
    # Pipeline status
    print("\n5. Pipeline Status:")
    status = pipeline.get_pipeline_status()
    
    for key, value in status.items():
        if key != 'last_update':
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n   ✓ Pipeline operational and monitoring continuously")

def main():
    """Run all continuous learning demos"""
    print("CONTINUOUS LEARNING PIPELINE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete continuous learning system for")
    print("flight delay prediction with online/offline parity, drift detection,")
    print("ensemble methods, model versioning, and incremental learning.")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_feature_engineering()
        demo_performance_monitoring()
        demo_ensemble_prediction()
        demo_model_versioning()
        demo_incremental_learning()
        demo_full_pipeline()
        
        print("\n" + "="*80)
        print("CONTINUOUS LEARNING DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Capabilities Demonstrated:")
        print("✓ Online/offline feature engineering parity")
        print("✓ Model performance monitoring and drift detection")
        print("✓ Automated retraining triggers")
        print("✓ Ensemble prediction methods")
        print("✓ Model versioning and deployment pipeline")
        print("✓ Incremental learning capabilities")
        print("✓ Complete pipeline orchestration")
        
        print("\nThe continuous learning pipeline provides:")
        print("• Real-time model adaptation to changing conditions")
        print("• Automated quality monitoring and drift detection")
        print("• Robust ensemble predictions with uncertainty quantification")
        print("• Safe model deployment with rollback capabilities")
        print("• Incremental learning for streaming data scenarios")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())