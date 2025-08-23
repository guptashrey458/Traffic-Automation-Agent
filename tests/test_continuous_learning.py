"""
Tests for Continuous Learning Pipeline

Tests cover:
- Feature engineering with online/offline parity
- Model performance monitoring and drift detection
- Automated retraining triggers
- Ensemble prediction methods
- Model versioning and deployment pipeline
- Incremental learning capabilities
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from src.services.continuous_learning import (
    ContinuousLearningPipeline,
    FeatureEngineer,
    ModelPerformanceMonitor,
    EnsemblePredictor,
    ModelVersionManager,
    IncrementalLearner,
    FeatureVector,
    ModelMetrics,
    DriftReport,
    EnsemblePrediction,
    ModelVersion
)

# Create a simple serializable model class at module level
class SimpleTestModel:
    """Simple test model that can be pickled"""
    def __init__(self, prediction_value=15.0):
        self.prediction_value = prediction_value
    
    def predict(self, X):
        return [self.prediction_value] * len(X)

class TestFeatureEngineer:
    """Test feature engineering with online/offline parity"""
    
    def setup_method(self):
        self.feature_engineer = FeatureEngineer()
    
    def test_extract_features_online(self):
        """Test online feature extraction"""
        flight_data = {
            'flight_id': 'AI123',
            'std_utc': '2024-01-15 08:30:00'
        }
        
        context_data = {
            'turnaround_time': 45.0,
            'runway_utilization': 0.8,
            'weather_score': 0.2,
            'inbound_delay': 10.0
        }
        
        features = self.feature_engineer.extract_features_online(flight_data, context_data)
        
        assert isinstance(features, FeatureVector)
        assert features.flight_id == 'AI123'
        assert features.source == 'online'
        assert 'hour_of_day' in features.features
        assert 'scheduled_turnaround_time' in features.features
        assert features.features['hour_of_day'] == 8.0
        assert features.features['scheduled_turnaround_time'] == 45.0
        assert len(features.feature_hash) == 32  # MD5 hash length
    
    def test_extract_features_offline(self):
        """Test offline feature extraction"""
        flight_df = pd.DataFrame({
            'flight_id': ['AI123', 'AI124'],
            'std_utc': ['2024-01-15 08:30:00', '2024-01-15 09:15:00']
        })
        
        features_list = self.feature_engineer.extract_features_offline(flight_df)
        
        assert len(features_list) == 2
        assert all(isinstance(fv, FeatureVector) for fv in features_list)
        assert all(fv.source == 'offline' for fv in features_list)
        assert features_list[0].features['hour_of_day'] == 8.0
        assert features_list[1].features['hour_of_day'] == 9.0
    
    def test_online_offline_parity(self):
        """Test that online and offline feature extraction produce consistent results"""
        flight_data = {
            'flight_id': 'AI123',
            'std_utc': '2024-01-15 08:30:00'
        }
        
        context_data = {
            'turnaround_time': 45.0,
            'runway_utilization': 0.8,
            'weather_score': 0.2,
            'inbound_delay': 10.0
        }
        
        # Online extraction
        online_features = self.feature_engineer.extract_features_online(flight_data, context_data)
        
        # Offline extraction
        flight_df = pd.DataFrame([flight_data])
        offline_features = self.feature_engineer.extract_features_offline(flight_df)[0]
        
        # Compare temporal features (should be identical)
        temporal_features = ['hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour']
        for feature in temporal_features:
            assert online_features.features[feature] == offline_features.features[feature]
    
    def test_fit_transformations(self):
        """Test fitting scaling transformations"""
        feature_vectors = []
        for i in range(10):
            features = FeatureVector(
                flight_id=f'AI{i}',
                timestamp=datetime.now(),
                features={'test_feature': float(i * 10)},
                feature_hash='test',
                source='test'
            )
            feature_vectors.append(features)
        
        self.feature_engineer.fit_transformations(feature_vectors)
        
        assert 'test_feature' in self.feature_engineer.scaler_params
        mean, std = self.feature_engineer.scaler_params['test_feature']
        assert abs(mean - 45.0) < 1e-6  # Mean of 0,10,20,...,90
        assert std > 0

class TestModelPerformanceMonitor:
    """Test model performance monitoring and drift detection"""
    
    def setup_method(self):
        self.monitor = ModelPerformanceMonitor(drift_threshold=0.1)
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        predictions = [10.0, 20.0, 5.0, 30.0, 15.0]
        actuals = [12.0, 18.0, 8.0, 25.0, 16.0]
        
        metrics = self.monitor.evaluate_model_performance('test_model', predictions, actuals)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.model_id == 'test_model'
        assert metrics.mae > 0
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.auc <= 1
        assert metrics.prediction_count == 5
        assert metrics.drift_score >= 0
        assert 0 <= metrics.confidence_score <= 1
    
    def test_detect_drift_no_baseline(self):
        """Test drift detection without baseline"""
        metrics = ModelMetrics(
            model_id='test',
            timestamp=datetime.now(),
            mae=5.0,
            accuracy=0.8,
            auc=0.7,
            prediction_count=100,
            drift_score=0.05,
            confidence_score=0.9
        )
        
        drift_report = self.monitor.detect_drift('test', metrics)
        
        assert isinstance(drift_report, DriftReport)
        assert not drift_report.drift_detected  # No baseline to compare
        assert drift_report.drift_score == 0.05
    
    def test_detect_drift_with_degradation(self):
        """Test drift detection with performance degradation"""
        # Set baseline
        baseline_predictions = [10.0, 20.0, 5.0, 30.0, 15.0]
        baseline_actuals = [12.0, 18.0, 8.0, 25.0, 16.0]
        self.monitor.evaluate_model_performance('test', baseline_predictions, baseline_actuals)
        
        # Simulate degraded performance
        degraded_predictions = [15.0, 25.0, 10.0, 35.0, 20.0]
        degraded_actuals = [12.0, 18.0, 8.0, 25.0, 16.0]
        degraded_metrics = self.monitor.evaluate_model_performance('test', degraded_predictions, degraded_actuals)
        
        drift_report = self.monitor.detect_drift('test', degraded_metrics)
        
        # Should detect drift due to increased MAE
        assert drift_report.drift_detected
        assert "degradation" in drift_report.recommendation.lower()
    
    def test_record_prediction(self):
        """Test prediction recording"""
        # This is a placeholder test since record_prediction doesn't do much currently
        self.monitor.record_prediction('test_model', 15.0, 12.0)
        # No assertion needed, just ensure no exception

class TestEnsemblePredictor:
    """Test ensemble prediction methods"""
    
    def setup_method(self):
        self.ensemble = EnsemblePredictor()
    
    def test_add_model(self):
        """Test adding models to ensemble"""
        mock_model = Mock()
        mock_model.predict.return_value = [15.0]
        
        self.ensemble.add_model('model1', mock_model, weight=0.6)
        self.ensemble.add_model('model2', mock_model, weight=0.4)
        
        assert 'model1' in self.ensemble.models
        assert 'model2' in self.ensemble.models
        assert self.ensemble.model_weights['model1'] == 0.6
        assert self.ensemble.model_weights['model2'] == 0.4
    
    def test_predict_ensemble(self):
        """Test ensemble prediction"""
        # Create mock models
        model1 = Mock()
        model1.predict.return_value = [10.0]
        
        model2 = Mock()
        model2.predict.return_value = [20.0]
        
        self.ensemble.add_model('model1', model1, weight=0.7)
        self.ensemble.add_model('model2', model2, weight=0.3)
        
        # Create test features
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0, 'feature2': 2.0},
            feature_hash='test',
            source='test'
        )
        
        prediction = self.ensemble.predict_ensemble(features)
        
        assert isinstance(prediction, EnsemblePrediction)
        # Weighted average: (10.0 * 0.7 + 20.0 * 0.3) / (0.7 + 0.3) = 13.0
        assert abs(prediction.prediction - 13.0) < 1e-6
        assert 'model1' in prediction.component_predictions
        assert 'model2' in prediction.component_predictions
        assert prediction.component_predictions['model1'] == 10.0
        assert prediction.component_predictions['model2'] == 20.0
        assert prediction.confidence > 0
        assert prediction.uncertainty >= 0
    
    def test_predict_ensemble_empty(self):
        """Test ensemble prediction with no models"""
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0},
            feature_hash='test',
            source='test'
        )
        
        prediction = self.ensemble.predict_ensemble(features)
        
        assert prediction.prediction == 0.0
        assert prediction.confidence == 0.0
        assert prediction.uncertainty == 1.0
    
    def test_update_weights(self):
        """Test updating model weights based on performance"""
        # Add models
        self.ensemble.add_model('model1', Mock(), weight=0.5)
        self.ensemble.add_model('model2', Mock(), weight=0.5)
        
        # Create performance metrics
        metrics1 = ModelMetrics('model1', datetime.now(), mae=5.0, accuracy=0.8, 
                               auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        metrics2 = ModelMetrics('model2', datetime.now(), mae=10.0, accuracy=0.7, 
                               auc=0.6, prediction_count=100, drift_score=0.08, confidence_score=0.8)
        
        performance_metrics = {'model1': metrics1, 'model2': metrics2}
        
        self.ensemble.update_weights(performance_metrics)
        
        # Model1 should have higher weight due to lower MAE
        assert self.ensemble.model_weights['model1'] > self.ensemble.model_weights['model2']
        # Weights should sum to 1
        assert abs(sum(self.ensemble.model_weights.values()) - 1.0) < 1e-6

class TestModelVersionManager:
    """Test model versioning and deployment pipeline"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(self.temp_dir)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_model_version(self):
        """Test saving model version"""
        mock_model = SimpleTestModel()
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        training_hash = 'test_hash'
        
        version_id = self.version_manager.save_model_version(
            'delay_predictor', mock_model, metrics, feature_importance, training_hash
        )
        
        assert version_id.startswith('delay_predictor_')
        assert version_id in self.version_manager.versions
        
        version = self.version_manager.versions[version_id]
        assert version.model_type == 'delay_predictor'
        assert version.training_data_hash == training_hash
        assert not version.is_active
        assert version.rollback_available
        
        # Check files exist
        model_path = Path(self.temp_dir) / f"{version_id}.pkl"
        metadata_path = Path(self.temp_dir) / f"{version_id}_metadata.json"
        assert model_path.exists()
        assert metadata_path.exists()
    
    def test_deploy_model(self):
        """Test model deployment"""
        mock_model = SimpleTestModel()
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        
        version_id = self.version_manager.save_model_version(
            'delay_predictor', mock_model, metrics, {}, 'hash'
        )
        
        # Deploy the model
        success = self.version_manager.deploy_model(version_id)
        
        assert success
        assert self.version_manager.versions[version_id].is_active
        assert self.version_manager.active_models['delay_predictor'] == version_id
    
    def test_rollback_model(self):
        """Test model rollback"""
        mock_model = SimpleTestModel()
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        
        version1 = self.version_manager.save_model_version(
            'delay_predictor', mock_model, metrics, {}, 'hash1'
        )
        
        # Wait a bit to ensure different timestamps
        import time
        time.sleep(0.1)  # Increase sleep time
        
        version2 = self.version_manager.save_model_version(
            'delay_predictor', mock_model, metrics, {}, 'hash2'
        )
        
        # Deploy the second version
        self.version_manager.deploy_model(version2)
        
        # Rollback to first version
        success = self.version_manager.rollback_model('delay_predictor')
        
        assert success
        assert self.version_manager.active_models['delay_predictor'] == version1
        assert self.version_manager.versions[version1].is_active
        assert not self.version_manager.versions[version2].is_active
    
    def test_load_active_model(self):
        """Test loading active model"""
        mock_model = SimpleTestModel()
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        
        version_id = self.version_manager.save_model_version(
            'delay_predictor', mock_model, metrics, {}, 'hash'
        )
        self.version_manager.deploy_model(version_id)
        
        # Load the active model
        loaded_model = self.version_manager.load_active_model('delay_predictor')
        
        assert loaded_model is not None
    
    def test_cleanup_old_versions(self):
        """Test cleanup of old model versions"""
        mock_model = SimpleTestModel()
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.05, confidence_score=0.9)
        
        versions = []
        for i in range(7):
            version_id = self.version_manager.save_model_version(
                'delay_predictor', mock_model, metrics, {}, f'hash{i}'
            )
            versions.append(version_id)
            import time
            time.sleep(0.01)  # Ensure different timestamps
        
        # Cleanup, keeping only 3 versions
        self.version_manager.cleanup_old_versions(keep_versions=3)
        
        # Should have only 3 versions left
        remaining_versions = [v for v in self.version_manager.versions.values() 
                            if v.model_type == 'delay_predictor']
        assert len(remaining_versions) == 3
        
        # Should keep the most recent ones
        remaining_ids = [v.version_id for v in remaining_versions]
        assert versions[-3] in remaining_ids
        assert versions[-2] in remaining_ids
        assert versions[-1] in remaining_ids

class TestIncrementalLearner:
    """Test incremental learning capabilities"""
    
    def setup_method(self):
        self.learner = IncrementalLearner(batch_size=3)
    
    def test_add_training_sample(self):
        """Test adding training samples"""
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0, 'feature2': 2.0},
            feature_hash='test',
            source='test'
        )
        
        self.learner.add_training_sample(features, 15.0)
        
        assert len(self.learner.update_buffer) == 1
        assert self.learner.update_buffer[0][0] == features
        assert self.learner.update_buffer[0][1] == 15.0
    
    @patch('src.services.continuous_learning.lgb.LGBMRegressor')
    def test_update_model(self, mock_lgb):
        """Test model update with buffered samples"""
        # Mock LightGBM model
        mock_model = Mock()
        mock_lgb.return_value = mock_model
        
        # Add samples to buffer
        for i in range(3):
            features = FeatureVector(
                flight_id=f'test{i}',
                timestamp=datetime.now(),
                features={'feature1': float(i), 'feature2': float(i*2)},
                feature_hash='test',
                source='test'
            )
            self.learner.add_training_sample(features, float(i * 10))
        
        # Buffer should be cleared after update
        assert len(self.learner.update_buffer) == 0
        
        # Model should be created and fitted
        mock_lgb.assert_called_once()
        mock_model.fit.assert_called_once()
        
        # Model should be stored
        assert self.learner.model == mock_model
    
    def test_predict(self):
        """Test prediction with incremental learner"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = [15.0]
        self.learner.model = mock_model
        
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0, 'feature2': 2.0},
            feature_hash='test',
            source='test'
        )
        
        prediction = self.learner.predict(features)
        
        assert prediction == 15.0
        mock_model.predict.assert_called_once()
    
    def test_predict_no_model(self):
        """Test prediction without trained model"""
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0, 'feature2': 2.0},
            feature_hash='test',
            source='test'
        )
        
        prediction = self.learner.predict(features)
        
        assert prediction == 0.0

class TestContinuousLearningPipeline:
    """Test main continuous learning pipeline"""
    
    def setup_method(self):
        self.pipeline = ContinuousLearningPipeline()
    
    def test_update_features_online(self):
        """Test online feature update"""
        flight_data = {
            'flight_id': 'AI123',
            'std_utc': '2024-01-15 08:30:00'
        }
        
        context_data = {
            'turnaround_time': 45.0,
            'runway_utilization': 0.8
        }
        
        features = self.pipeline.update_features_online(flight_data, context_data)
        
        assert isinstance(features, FeatureVector)
        assert features.flight_id == 'AI123'
        assert features.source == 'online'
    
    def test_predict_with_ensemble(self):
        """Test ensemble prediction"""
        # Add a mock model to ensemble
        mock_model = Mock()
        mock_model.predict.return_value = [15.0]
        self.pipeline.ensemble_predictor.add_model('test_model', mock_model)
        
        features = FeatureVector(
            flight_id='test',
            timestamp=datetime.now(),
            features={'feature1': 1.0, 'feature2': 2.0},
            feature_hash='test',
            source='test'
        )
        
        prediction = self.pipeline.predict_with_ensemble(features)
        
        assert isinstance(prediction, EnsemblePrediction)
        assert prediction.prediction == 15.0
    
    def test_monitor_model_drift(self):
        """Test model drift monitoring"""
        predictions = [10.0, 20.0, 5.0, 30.0, 15.0]
        actuals = [12.0, 18.0, 8.0, 25.0, 16.0]
        
        drift_report = self.pipeline.monitor_model_drift(predictions, actuals)
        
        assert isinstance(drift_report, DriftReport)
        assert drift_report.model_id == 'ensemble'
    
    def test_trigger_retraining(self):
        """Test retraining trigger logic"""
        # No history - should not trigger
        assert not self.pipeline.trigger_retraining(0.1)
        
        # Add some performance history
        metrics = ModelMetrics('test', datetime.now(), mae=5.0, accuracy=0.8, 
                              auc=0.7, prediction_count=100, drift_score=0.15, confidence_score=0.9)
        self.pipeline.performance_monitor.performance_history.append(metrics)
        
        # Should trigger due to high drift score
        assert self.pipeline.trigger_retraining(0.1)
    
    def test_get_pipeline_status(self):
        """Test pipeline status reporting"""
        status = self.pipeline.get_pipeline_status()
        
        assert 'active_models' in status
        assert 'total_versions' in status
        assert 'performance_history_length' in status
        assert 'ensemble_models' in status
        assert 'incremental_buffer_size' in status
        assert 'last_update' in status
        
        assert isinstance(status['active_models'], dict)
        assert isinstance(status['total_versions'], int)
        assert isinstance(status['performance_history_length'], int)
        assert isinstance(status['ensemble_models'], list)
        assert isinstance(status['incremental_buffer_size'], int)
    
    @patch('src.services.continuous_learning.lgb.LGBMRegressor')
    def test_retrain_models_incremental(self, mock_lgb):
        """Test incremental model retraining"""
        # Mock LightGBM
        mock_model = Mock()
        mock_model.predict.return_value = np.array([10.0, 15.0])
        mock_model.feature_importances_ = np.array([0.6, 0.4])
        mock_lgb.return_value = mock_model
        
        # Create test data
        new_data = pd.DataFrame({
            'flight_id': ['AI123', 'AI124'],
            'std_utc': ['2024-01-15 08:30:00', '2024-01-15 09:15:00']
        })
        
        result = self.pipeline.retrain_models_incremental(new_data)
        
        assert result['success']
        assert isinstance(result['models_updated'], list)
        assert isinstance(result['performance_improvement'], dict)
        assert isinstance(result['errors'], list)

if __name__ == '__main__':
    pytest.main([__file__])