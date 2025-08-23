"""Tests for the analytics engine and peak detection algorithms."""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from src.services.analytics import (
    AnalyticsEngine, WeatherRegime, TrafficLevel, TimeBucket, 
    OverloadWindow, PeakAnalysis, CapacityConfig
)
from src.services.database import FlightDatabaseService, QueryResult


class TestTimeBucket:
    """Test TimeBucket functionality."""
    
    def test_time_bucket_initialization(self):
        """Test TimeBucket initialization and calculations."""
        start_time = datetime(2024, 1, 15, 8, 0)
        end_time = datetime(2024, 1, 15, 8, 10)
        
        bucket = TimeBucket(
            start_time=start_time,
            end_time=end_time,
            bucket_minutes=10,
            scheduled_departures=8,
            scheduled_arrivals=6,
            capacity=12
        )
        
        assert bucket.total_demand == 14
        assert bucket.utilization == pytest.approx(14/12, rel=1e-2)
        assert bucket.overload == 2
        assert bucket.traffic_level == TrafficLevel.HIGH
    
    def test_traffic_level_classification(self):
        """Test traffic level classification logic."""
        start_time = datetime(2024, 1, 15, 8, 0)
        end_time = datetime(2024, 1, 15, 8, 10)
        
        # Test LOW traffic
        bucket_low = TimeBucket(
            start_time=start_time, end_time=end_time, bucket_minutes=10,
            scheduled_departures=3, scheduled_arrivals=2, capacity=10
        )
        assert bucket_low.traffic_level == TrafficLevel.LOW
        
        # Test MEDIUM traffic
        bucket_medium = TimeBucket(
            start_time=start_time, end_time=end_time, bucket_minutes=10,
            scheduled_departures=4, scheduled_arrivals=3, capacity=10
        )
        assert bucket_medium.traffic_level == TrafficLevel.MEDIUM
        
        # Test HIGH traffic
        bucket_high = TimeBucket(
            start_time=start_time, end_time=end_time, bucket_minutes=10,
            scheduled_departures=6, scheduled_arrivals=4, capacity=10
        )
        assert bucket_high.traffic_level == TrafficLevel.HIGH
        
        # Test CRITICAL traffic
        bucket_critical = TimeBucket(
            start_time=start_time, end_time=end_time, bucket_minutes=10,
            scheduled_departures=8, scheduled_arrivals=5, capacity=10
        )
        assert bucket_critical.traffic_level == TrafficLevel.CRITICAL


class TestOverloadWindow:
    """Test OverloadWindow functionality."""
    
    def test_overload_window_severity_classification(self):
        """Test severity classification and recommendations."""
        start_time = datetime(2024, 1, 15, 8, 0)
        end_time = datetime(2024, 1, 15, 8, 30)
        
        # Test minor overload
        window_minor = OverloadWindow(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=30,
            peak_overload=3,
            avg_overload=2.5,
            affected_flights=25
        )
        assert window_minor.severity == "minor"
        assert "Monitor traffic flow" in window_minor.recommendations
        
        # Test moderate overload
        window_moderate = OverloadWindow(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=30,
            peak_overload=7,
            avg_overload=5.5,
            affected_flights=45
        )
        assert window_moderate.severity == "moderate"
        assert "Optimize runway usage" in window_moderate.recommendations
        
        # Test severe overload
        window_severe = OverloadWindow(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=30,
            peak_overload=12,
            avg_overload=10.0,
            affected_flights=65
        )
        assert window_severe.severity == "severe"
        assert "Consider ground delay program" in window_severe.recommendations


class TestCapacityConfig:
    """Test CapacityConfig functionality."""
    
    def test_default_capacity_initialization(self):
        """Test default capacity values for major airports."""
        # Test BOM (Mumbai) configuration
        config_bom = CapacityConfig("BOM")
        assert config_bom.runway_capacity_per_hour["total"] == 50
        assert config_bom.weather_adjustments[WeatherRegime.CALM] == 1.0
        assert config_bom.weather_adjustments[WeatherRegime.STRONG] == 0.65
        
        # Test DEL (Delhi) configuration
        config_del = CapacityConfig("DEL")
        assert config_del.runway_capacity_per_hour["total"] == 50
        
        # Test other airport configuration
        config_other = CapacityConfig("CCU")
        assert config_other.runway_capacity_per_hour["total"] == 35
    
    def test_capacity_adjustments(self):
        """Test capacity adjustments for weather and time of day."""
        config = CapacityConfig("BOM")
        
        # Test normal conditions
        normal_capacity = config.get_capacity(hour=10, weather=WeatherRegime.CALM)
        assert normal_capacity == 50  # Full capacity during day
        
        # Test weather impact
        stormy_capacity = config.get_capacity(hour=10, weather=WeatherRegime.STRONG)
        assert stormy_capacity == int(50 * 0.65)  # 65% capacity
        
        # Test night time impact
        night_capacity = config.get_capacity(hour=2, weather=WeatherRegime.CALM)
        assert night_capacity < 50  # Reduced capacity at night
        
        # Test curfew impact
        curfew_capacity = config.get_capacity(hour=3, weather=WeatherRegime.CALM)
        assert curfew_capacity < 10  # Severely restricted during curfew
    
    def test_operation_type_capacity(self):
        """Test capacity for different operation types."""
        config = CapacityConfig("BOM")
        
        total_cap = config.get_capacity(hour=10, operation_type="total")
        dep_cap = config.get_capacity(hour=10, operation_type="departures")
        arr_cap = config.get_capacity(hour=10, operation_type="arrivals")
        
        assert total_cap == 50
        assert dep_cap == 30
        assert arr_cap == 30


class TestAnalyticsEngine:
    """Test AnalyticsEngine functionality."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service."""
        mock_db = Mock(spec=FlightDatabaseService)
        return mock_db
    
    @pytest.fixture
    def analytics_engine(self, mock_db_service):
        """Create analytics engine with mock database."""
        return AnalyticsEngine(db_service=mock_db_service)
    
    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data for testing."""
        base_date = date(2024, 1, 15)
        return [
            {
                'flight_id': 'AI2509_001',
                'flight_number': 'AI2509',
                'origin_code': 'BOM',
                'destination_code': 'DEL',
                'std_utc': datetime.combine(base_date, time(8, 0)),
                'atd_utc': datetime.combine(base_date, time(8, 15)),
                'sta_utc': datetime.combine(base_date, time(10, 30)),
                'ata_utc': datetime.combine(base_date, time(10, 45)),
                'dep_delay_min': 15,
                'arr_delay_min': 15
            },
            {
                'flight_id': '6E123_001',
                'flight_number': '6E123',
                'origin_code': 'BOM',
                'destination_code': 'BLR',
                'std_utc': datetime.combine(base_date, time(8, 5)),
                'atd_utc': datetime.combine(base_date, time(8, 25)),
                'sta_utc': datetime.combine(base_date, time(9, 30)),
                'ata_utc': datetime.combine(base_date, time(9, 50)),
                'dep_delay_min': 20,
                'arr_delay_min': 20
            },
            {
                'flight_id': 'UK955_001',
                'flight_number': 'UK955',
                'origin_code': 'DEL',
                'destination_code': 'BOM',
                'std_utc': datetime.combine(base_date, time(8, 10)),
                'atd_utc': datetime.combine(base_date, time(8, 10)),
                'sta_utc': datetime.combine(base_date, time(8, 5)),  # Arrives at BOM at 8:05
                'ata_utc': datetime.combine(base_date, time(8, 5)),
                'dep_delay_min': 0,
                'arr_delay_min': 0
            }
        ]
    
    def test_analytics_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization."""
        assert analytics_engine.db_service is not None
        assert "BOM" in analytics_engine.capacity_configs
        assert "DEL" in analytics_engine.capacity_configs
        assert "DEFAULT" in analytics_engine.capacity_configs
    
    def test_get_flight_data(self, analytics_engine, sample_flight_data):
        """Test flight data retrieval."""
        # Mock the database query
        analytics_engine.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=sample_flight_data,
            row_count=len(sample_flight_data)
        )
        
        result = analytics_engine._get_flight_data("BOM", date(2024, 1, 15))
        
        assert len(result) == 3
        assert result[0]['flight_number'] == 'AI2509'
        analytics_engine.db_service.query_flights_by_date_range.assert_called_once()
    
    def test_create_time_buckets(self, analytics_engine, sample_flight_data):
        """Test time bucket creation and population."""
        analysis_date = date(2024, 1, 15)
        bucket_minutes = 10
        
        buckets = analytics_engine._create_time_buckets(
            sample_flight_data, analysis_date, bucket_minutes, "BOM", WeatherRegime.CALM
        )
        
        # Should create buckets for entire day
        assert len(buckets) == 24 * 60 // bucket_minutes  # 144 buckets for 10-minute intervals
        
        # Check first bucket
        first_bucket = buckets[0]
        assert first_bucket.start_time.hour == 0
        assert first_bucket.start_time.minute == 0
        assert first_bucket.bucket_minutes == bucket_minutes
        
        # Find bucket containing our test flights (8:00-8:10)
        target_bucket = None
        for bucket in buckets:
            if bucket.start_time.hour == 8 and bucket.start_time.minute == 0:
                target_bucket = bucket
                break
        
        assert target_bucket is not None
        assert target_bucket.scheduled_departures == 2  # AI2509 and 6E123 depart from BOM
        assert target_bucket.scheduled_arrivals == 1    # UK955 arrives at BOM at 8:05
    
    def test_populate_bucket_with_flights(self, analytics_engine, sample_flight_data):
        """Test flight population in time buckets."""
        start_time = datetime(2024, 1, 15, 8, 0)
        end_time = datetime(2024, 1, 15, 8, 10)
        
        bucket = TimeBucket(
            start_time=start_time,
            end_time=end_time,
            bucket_minutes=10,
            capacity=10
        )
        
        analytics_engine._populate_bucket_with_flights(bucket, sample_flight_data, "BOM")
        
        assert bucket.scheduled_departures == 2  # AI2509 and 6E123
        assert bucket.scheduled_arrivals == 1    # UK955 arrives at 8:05
        assert bucket.actual_departures == 2
        assert bucket.actual_arrivals == 1
        assert bucket.delayed_flights == 1  # Only 6E123 has >15min delay (20min), AI2509 has exactly 15min
        assert bucket.avg_delay > 0
    
    def test_identify_overload_windows(self, analytics_engine):
        """Test overload window identification."""
        # Create test buckets with overload pattern
        base_time = datetime(2024, 1, 15, 8, 0)
        buckets = []
        
        # Normal bucket
        buckets.append(TimeBucket(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=10),
            bucket_minutes=10,
            scheduled_departures=5, scheduled_arrivals=3, capacity=10
        ))
        
        # Overload bucket 1
        buckets.append(TimeBucket(
            start_time=base_time + timedelta(minutes=10),
            end_time=base_time + timedelta(minutes=20),
            bucket_minutes=10,
            scheduled_departures=8, scheduled_arrivals=5, capacity=10
        ))
        
        # Overload bucket 2 (continues overload)
        buckets.append(TimeBucket(
            start_time=base_time + timedelta(minutes=20),
            end_time=base_time + timedelta(minutes=30),
            bucket_minutes=10,
            scheduled_departures=7, scheduled_arrivals=6, capacity=10
        ))
        
        # Normal bucket (ends overload)
        buckets.append(TimeBucket(
            start_time=base_time + timedelta(minutes=30),
            end_time=base_time + timedelta(minutes=40),
            bucket_minutes=10,
            scheduled_departures=4, scheduled_arrivals=3, capacity=10
        ))
        
        overload_windows = analytics_engine._identify_overload_windows(buckets)
        
        assert len(overload_windows) == 1
        window = overload_windows[0]
        assert window.duration_minutes == 20
        assert window.peak_overload == 3  # Max overload from the buckets
        assert window.start_time == base_time + timedelta(minutes=10)
    
    def test_find_peak_hour(self, analytics_engine):
        """Test peak hour identification."""
        base_time = datetime(2024, 1, 15, 0, 0)
        buckets = []
        
        # Create buckets with varying demand throughout the day
        for hour in range(24):
            for minute in [0, 10, 20, 30, 40, 50]:
                bucket_time = base_time + timedelta(hours=hour, minutes=minute)
                
                # Simulate higher demand during morning rush (8-10 AM)
                if 8 <= hour <= 9:
                    demand = 15
                elif 6 <= hour <= 7 or 10 <= hour <= 11:
                    demand = 10
                else:
                    demand = 5
                
                bucket = TimeBucket(
                    start_time=bucket_time,
                    end_time=bucket_time + timedelta(minutes=10),
                    bucket_minutes=10,
                    scheduled_departures=demand//2,
                    scheduled_arrivals=demand//2,
                    capacity=10
                )
                buckets.append(bucket)
        
        peak_hour, peak_demand = analytics_engine._find_peak_hour(buckets)
        
        assert peak_hour in [8, 9]  # Should be morning rush hour
        assert peak_demand > 60  # 6 buckets * 15 demand each = 90
    
    def test_identify_delay_hotspots(self, analytics_engine):
        """Test delay hotspot identification."""
        base_time = datetime(2024, 1, 15, 8, 0)
        buckets = []
        
        # Create buckets with varying delay patterns
        for i in range(6):
            bucket_time = base_time + timedelta(minutes=i*10)
            
            # Create high delay in middle buckets
            if 2 <= i <= 3:
                avg_delay = 35.0
                delayed_flights = 8
                total_demand = 12
            else:
                avg_delay = 5.0
                delayed_flights = 1
                total_demand = 8
            
            bucket = TimeBucket(
                start_time=bucket_time,
                end_time=bucket_time + timedelta(minutes=10),
                bucket_minutes=10,
                scheduled_departures=total_demand//2,
                scheduled_arrivals=total_demand//2,
                capacity=10
            )
            bucket.avg_delay = avg_delay
            bucket.delayed_flights = delayed_flights
            bucket.total_demand = total_demand
            
            buckets.append(bucket)
        
        hotspots = analytics_engine._identify_delay_hotspots(buckets)
        
        assert len(hotspots) == 2  # Two high-delay buckets
        assert hotspots[0]["avg_delay"] == 35.0  # Sorted by delay severity
        assert hotspots[0]["severity"] == "high"
        assert hotspots[1]["avg_delay"] == 35.0
    
    def test_generate_recommendations(self, analytics_engine):
        """Test recommendation generation."""
        # Create scenario with overload windows
        overload_windows = [
            OverloadWindow(
                start_time=datetime(2024, 1, 15, 8, 0),
                end_time=datetime(2024, 1, 15, 8, 30),
                duration_minutes=30,
                peak_overload=12,  # Severe overload
                avg_overload=10.0,
                affected_flights=50
            )
        ]
        
        # Create high utilization buckets
        base_time = datetime(2024, 1, 15, 8, 0)
        high_util_buckets = []
        for i in range(8):  # More than 1 hour of high utilization
            bucket = TimeBucket(
                start_time=base_time + timedelta(minutes=i*10),
                end_time=base_time + timedelta(minutes=(i+1)*10),
                bucket_minutes=10,
                scheduled_departures=9,
                scheduled_arrivals=9,
                capacity=10
            )
            high_util_buckets.append(bucket)
        
        recommendations = analytics_engine._generate_recommendations(
            high_util_buckets, overload_windows, WeatherRegime.STRONG
        )
        
        assert len(recommendations) > 0
        assert any("ground delay program" in rec.lower() for rec in recommendations)
        assert any("weather contingency" in rec.lower() for rec in recommendations)
        assert any("spreading traffic" in rec.lower() for rec in recommendations)
    
    def test_analyze_peaks_integration(self, analytics_engine, sample_flight_data):
        """Test complete peak analysis integration."""
        # Mock the database query
        analytics_engine.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=sample_flight_data,
            row_count=len(sample_flight_data)
        )
        
        analysis = analytics_engine.analyze_peaks(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10,
            weather_regime=WeatherRegime.CALM
        )
        
        assert analysis.airport == "BOM"
        assert analysis.analysis_date == date(2024, 1, 15)
        assert analysis.bucket_minutes == 10
        assert analysis.weather_regime == WeatherRegime.CALM
        assert len(analysis.time_buckets) == 144  # 24 hours * 6 buckets per hour
        assert analysis.peak_hour is not None
        assert len(analysis.recommendations) > 0
    
    def test_analyze_peaks_no_data(self, analytics_engine):
        """Test peak analysis with no flight data."""
        # Mock empty database response
        analytics_engine.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=[],
            row_count=0
        )
        
        analysis = analytics_engine.analyze_peaks(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10
        )
        
        assert analysis.airport == "BOM"
        assert len(analysis.time_buckets) == 0
        assert "No flight data available" in analysis.recommendations[0]
    
    def test_generate_demand_heatmap(self, analytics_engine, sample_flight_data):
        """Test demand heatmap generation."""
        # Mock the database query
        analytics_engine.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=sample_flight_data,
            row_count=len(sample_flight_data)
        )
        
        heatmap = analytics_engine.generate_demand_heatmap(
            airport="BOM",
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
            bucket_minutes=10
        )
        
        assert heatmap["airport"] == "BOM"
        assert heatmap["bucket_minutes"] == 10
        assert len(heatmap["heatmap_data"]) == 144  # 24 hours * 6 buckets
        assert len(heatmap["date_range"]) == 1
        assert "summary" in heatmap
        assert heatmap["summary"]["total_buckets"] == 144
    
    def test_capacity_config_management(self, analytics_engine):
        """Test capacity configuration management."""
        # Test getting default config
        config = analytics_engine.get_capacity_config("BOM")
        assert config.airport_code == "BOM"
        
        # Test updating config
        new_config = CapacityConfig(
            airport_code="CCU",
            runway_capacity_per_hour={"total": 40, "departures": 25, "arrivals": 25}
        )
        analytics_engine.update_capacity_config("CCU", new_config)
        
        retrieved_config = analytics_engine.get_capacity_config("CCU")
        assert retrieved_config.runway_capacity_per_hour["total"] == 40
    
    def test_peak_analysis_heatmap_data(self, analytics_engine, sample_flight_data):
        """Test heatmap data generation from peak analysis."""
        # Mock the database query
        analytics_engine.db_service.query_flights_by_date_range.return_value = QueryResult(
            data=sample_flight_data,
            row_count=len(sample_flight_data)
        )
        
        analysis = analytics_engine.analyze_peaks(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10
        )
        
        heatmap_data = analysis.get_heatmap_data()
        
        assert len(heatmap_data) == 144
        
        # Check data structure
        sample_bucket = heatmap_data[0]
        required_fields = [
            "time", "hour", "minute", "demand", "capacity", 
            "utilization", "overload", "traffic_level", 
            "avg_delay", "delayed_flights", "color"
        ]
        
        for field in required_fields:
            assert field in sample_bucket
        
        # Check that 8:00 AM bucket has our test data
        bucket_8am = next((b for b in heatmap_data if b["hour"] == 8 and b["minute"] == 0), None)
        assert bucket_8am is not None
        assert bucket_8am["demand"] == 3  # 2 departures + 1 arrival


if __name__ == "__main__":
    pytest.main([__file__])