"""Integration tests for analytics engine with real data scenarios."""

import pytest
from datetime import datetime, date, time
from unittest.mock import Mock

from src.services.analytics import AnalyticsEngine, WeatherRegime
from src.services.database import FlightDatabaseService, QueryResult


class TestAnalyticsIntegration:
    """Integration tests for analytics engine."""
    
    @pytest.fixture
    def realistic_flight_data(self):
        """Create realistic flight data for Mumbai airport."""
        base_date = date(2024, 1, 15)
        flights = []
        
        # Morning rush hour flights (6 AM - 10 AM)
        morning_flights = [
            # 6 AM slot - Early morning departures
            {'flight_id': 'AI101', 'flight_number': 'AI101', 'origin_code': 'BOM', 'destination_code': 'DEL',
             'std_utc': datetime.combine(base_date, time(6, 0)), 'atd_utc': datetime.combine(base_date, time(6, 5)),
             'dep_delay_min': 5, 'arr_delay_min': None},
            {'flight_id': '6E201', 'flight_number': '6E201', 'origin_code': 'BOM', 'destination_code': 'BLR',
             'std_utc': datetime.combine(base_date, time(6, 10)), 'atd_utc': datetime.combine(base_date, time(6, 15)),
             'dep_delay_min': 5, 'arr_delay_min': None},
            
            # 7 AM slot - Building traffic
            {'flight_id': 'AI201', 'flight_number': 'AI201', 'origin_code': 'BOM', 'destination_code': 'CCU',
             'std_utc': datetime.combine(base_date, time(7, 0)), 'atd_utc': datetime.combine(base_date, time(7, 10)),
             'dep_delay_min': 10, 'arr_delay_min': None},
            {'flight_id': '6E301', 'flight_number': '6E301', 'origin_code': 'BOM', 'destination_code': 'HYD',
             'std_utc': datetime.combine(base_date, time(7, 15)), 'atd_utc': datetime.combine(base_date, time(7, 30)),
             'dep_delay_min': 15, 'arr_delay_min': None},
            {'flight_id': 'UK801', 'flight_number': 'UK801', 'origin_code': 'BOM', 'destination_code': 'GOI',
             'std_utc': datetime.combine(base_date, time(7, 30)), 'atd_utc': datetime.combine(base_date, time(7, 45)),
             'dep_delay_min': 15, 'arr_delay_min': None},
            
            # 8 AM slot - Peak hour with overload (all within 8:00-8:10 bucket)
            {'flight_id': 'AI301', 'flight_number': 'AI301', 'origin_code': 'BOM', 'destination_code': 'DEL',
             'std_utc': datetime.combine(base_date, time(8, 0)), 'atd_utc': datetime.combine(base_date, time(8, 25)),
             'dep_delay_min': 25, 'arr_delay_min': None},
            {'flight_id': '6E401', 'flight_number': '6E401', 'origin_code': 'BOM', 'destination_code': 'BLR',
             'std_utc': datetime.combine(base_date, time(8, 2)), 'atd_utc': datetime.combine(base_date, time(8, 35)),
             'dep_delay_min': 33, 'arr_delay_min': None},
            {'flight_id': 'SG501', 'flight_number': 'SG501', 'origin_code': 'BOM', 'destination_code': 'MAA',
             'std_utc': datetime.combine(base_date, time(8, 4)), 'atd_utc': datetime.combine(base_date, time(8, 40)),
             'dep_delay_min': 36, 'arr_delay_min': None},
            {'flight_id': 'AI401', 'flight_number': 'AI401', 'origin_code': 'BOM', 'destination_code': 'CCU',
             'std_utc': datetime.combine(base_date, time(8, 6)), 'atd_utc': datetime.combine(base_date, time(8, 50)),
             'dep_delay_min': 44, 'arr_delay_min': None},
            {'flight_id': '6E501', 'flight_number': '6E501', 'origin_code': 'BOM', 'destination_code': 'AMD',
             'std_utc': datetime.combine(base_date, time(8, 8)), 'atd_utc': datetime.combine(base_date, time(8, 55)),
             'dep_delay_min': 47, 'arr_delay_min': None},
            {'flight_id': 'UK901', 'flight_number': 'UK901', 'origin_code': 'BOM', 'destination_code': 'PNQ',
             'std_utc': datetime.combine(base_date, time(8, 9)), 'atd_utc': datetime.combine(base_date, time(9, 0)),
             'dep_delay_min': 51, 'arr_delay_min': None},
            
            # Add more flights to create overload
            {'flight_id': 'AI501', 'flight_number': 'AI501', 'origin_code': 'BOM', 'destination_code': 'JAI',
             'std_utc': datetime.combine(base_date, time(8, 1)), 'atd_utc': datetime.combine(base_date, time(8, 35)),
             'dep_delay_min': 34, 'arr_delay_min': None},
            {'flight_id': '6E601', 'flight_number': '6E601', 'origin_code': 'BOM', 'destination_code': 'IXC',
             'std_utc': datetime.combine(base_date, time(8, 3)), 'atd_utc': datetime.combine(base_date, time(8, 40)),
             'dep_delay_min': 37, 'arr_delay_min': None},
            
            # Arrivals during peak hour (all within 8:00-8:10 bucket)
            {'flight_id': 'AI501', 'flight_number': 'AI501', 'origin_code': 'DEL', 'destination_code': 'BOM',
             'std_utc': datetime.combine(base_date, time(6, 0)), 'atd_utc': datetime.combine(base_date, time(6, 0)),
             'sta_utc': datetime.combine(base_date, time(8, 0)), 'ata_utc': datetime.combine(base_date, time(8, 15)),
             'dep_delay_min': 0, 'arr_delay_min': 15},
            {'flight_id': '6E601', 'flight_number': '6E601', 'origin_code': 'BLR', 'destination_code': 'BOM',
             'std_utc': datetime.combine(base_date, time(6, 30)), 'atd_utc': datetime.combine(base_date, time(6, 30)),
             'sta_utc': datetime.combine(base_date, time(8, 3)), 'ata_utc': datetime.combine(base_date, time(8, 18)),
             'dep_delay_min': 0, 'arr_delay_min': 15},
            {'flight_id': 'SG601', 'flight_number': 'SG601', 'origin_code': 'MAA', 'destination_code': 'BOM',
             'std_utc': datetime.combine(base_date, time(6, 15)), 'atd_utc': datetime.combine(base_date, time(6, 15)),
             'sta_utc': datetime.combine(base_date, time(8, 7)), 'ata_utc': datetime.combine(base_date, time(8, 22)),
             'dep_delay_min': 0, 'arr_delay_min': 15},
        ]
        
        return morning_flights
    
    def test_peak_analysis_realistic_scenario(self, realistic_flight_data):
        """Test peak analysis with realistic Mumbai morning rush scenario."""
        # Create analytics engine with mock database
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=realistic_flight_data,
            row_count=len(realistic_flight_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        # Analyze peaks for Mumbai on a busy morning
        analysis = analytics_engine.analyze_peaks(
            airport="BOM",
            analysis_date=date(2024, 1, 15),
            bucket_minutes=10,
            weather_regime=WeatherRegime.CALM
        )
        
        # Verify basic analysis results
        assert analysis.airport == "BOM"
        assert analysis.analysis_date == date(2024, 1, 15)
        assert analysis.bucket_minutes == 10
        assert len(analysis.time_buckets) == 144  # 24 hours * 6 buckets per hour
        
        # Find the 8:00-8:10 AM bucket (peak hour)
        peak_bucket = None
        for bucket in analysis.time_buckets:
            if bucket.start_time.hour == 8 and bucket.start_time.minute == 0:
                peak_bucket = bucket
                break
        
        assert peak_bucket is not None
        assert peak_bucket.scheduled_departures == 8  # 8 departures scheduled in 8:00-8:10
        assert peak_bucket.scheduled_arrivals == 3    # 3 arrivals scheduled in 8:00-8:10
        assert peak_bucket.total_demand == 11         # Total demand
        assert peak_bucket.overload == 1              # 1 flight over capacity (11-10)
        assert peak_bucket.traffic_level.value in ["high", "critical"]  # Overloaded
        
        # Verify overload windows are detected
        assert len(analysis.overload_windows) > 0
        
        # Check that peak hour is identified correctly
        assert analysis.peak_hour == 8  # 8 AM should be the peak hour
        
        # Verify delay hotspots are identified
        assert len(analysis.delay_hotspots) > 0
        
        # Check recommendations are generated
        assert len(analysis.recommendations) > 0
        assert any("overload" in rec.lower() or "delay" in rec.lower() for rec in analysis.recommendations)
    
    def test_weather_impact_on_capacity(self, realistic_flight_data):
        """Test how weather conditions affect capacity and analysis."""
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=realistic_flight_data,
            row_count=len(realistic_flight_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        # Analyze under different weather conditions
        calm_analysis = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15),
            bucket_minutes=10, weather_regime=WeatherRegime.CALM
        )
        
        strong_wind_analysis = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15),
            bucket_minutes=10, weather_regime=WeatherRegime.STRONG
        )
        
        # Find peak buckets for comparison
        calm_peak = next(b for b in calm_analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
        strong_peak = next(b for b in strong_wind_analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
        
        # Strong winds should reduce capacity
        assert strong_peak.capacity < calm_peak.capacity
        
        # Strong winds should increase overload for same demand
        assert strong_peak.overload >= calm_peak.overload
        
        # Strong winds should trigger weather-specific recommendations
        weather_recs = [rec for rec in strong_wind_analysis.recommendations if "weather" in rec.lower()]
        assert len(weather_recs) > 0
    
    def test_different_bucket_sizes(self, realistic_flight_data):
        """Test analysis with different time bucket sizes."""
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=realistic_flight_data,
            row_count=len(realistic_flight_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        # Test 5-minute buckets
        analysis_5min = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=5
        )
        
        # Test 15-minute buckets
        analysis_15min = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=15
        )
        
        # Test 30-minute buckets
        analysis_30min = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=30
        )
        
        # Verify bucket counts
        assert len(analysis_5min.time_buckets) == 24 * 12   # 288 buckets
        assert len(analysis_15min.time_buckets) == 24 * 4   # 96 buckets
        assert len(analysis_30min.time_buckets) == 24 * 2   # 48 buckets
        
        # Smaller buckets should show more granular overload patterns
        assert len(analysis_5min.overload_windows) >= len(analysis_30min.overload_windows)
    
    def test_heatmap_generation_multi_day(self, realistic_flight_data):
        """Test heatmap generation across multiple days."""
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=realistic_flight_data,
            row_count=len(realistic_flight_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        # Generate heatmap for 3 days
        heatmap = analytics_engine.generate_demand_heatmap(
            airport="BOM",
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 17),
            bucket_minutes=10
        )
        
        # Verify heatmap structure
        assert heatmap["airport"] == "BOM"
        assert heatmap["bucket_minutes"] == 10
        assert len(heatmap["date_range"]) == 3  # 3 days
        assert len(heatmap["heatmap_data"]) == 3 * 144  # 3 days * 144 buckets per day
        
        # Verify summary statistics
        assert "summary" in heatmap
        assert heatmap["summary"]["total_demand"] > 0
        assert heatmap["summary"]["avg_utilization"] >= 0
        assert heatmap["summary"]["peak_utilization"] >= heatmap["summary"]["avg_utilization"]
    
    def test_capacity_configuration_customization(self, realistic_flight_data):
        """Test custom capacity configuration."""
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=realistic_flight_data,
            row_count=len(realistic_flight_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        # Get default analysis
        default_analysis = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=10
        )
        
        # Create custom capacity config with higher capacity
        from src.services.analytics import CapacityConfig
        custom_config = CapacityConfig(
            airport_code="BOM",
            runway_capacity_per_hour={
                "departures": 50,  # Increased from 35
                "arrivals": 50,    # Increased from 35
                "total": 80        # Increased from 60
            }
        )
        
        analytics_engine.update_capacity_config("BOM", custom_config)
        
        # Get analysis with custom config
        custom_analysis = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=10
        )
        
        # Find peak buckets
        default_peak = next(b for b in default_analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
        custom_peak = next(b for b in custom_analysis.time_buckets if b.start_time.hour == 8 and b.start_time.minute == 0)
        
        # Custom config should have higher capacity
        assert custom_peak.capacity > default_peak.capacity
        
        # Custom config should have lower overload for same demand
        assert custom_peak.overload <= default_peak.overload
    
    def test_empty_time_periods(self):
        """Test analysis with periods of no flight activity."""
        # Create data with flights only in morning, leaving afternoon/evening empty
        sparse_data = [
            {'flight_id': 'AI101', 'flight_number': 'AI101', 'origin_code': 'BOM', 'destination_code': 'DEL',
             'std_utc': datetime(2024, 1, 15, 6, 0), 'atd_utc': datetime(2024, 1, 15, 6, 5),
             'dep_delay_min': 5, 'arr_delay_min': None},
            {'flight_id': 'AI201', 'flight_number': 'AI201', 'origin_code': 'BOM', 'destination_code': 'BLR',
             'std_utc': datetime(2024, 1, 15, 7, 0), 'atd_utc': datetime(2024, 1, 15, 7, 5),
             'dep_delay_min': 5, 'arr_delay_min': None},
        ]
        
        mock_db = Mock(spec=FlightDatabaseService)
        mock_db.query_flights_by_date_range.return_value = QueryResult(
            data=sparse_data,
            row_count=len(sparse_data)
        )
        
        analytics_engine = AnalyticsEngine(db_service=mock_db)
        
        analysis = analytics_engine.analyze_peaks(
            airport="BOM", analysis_date=date(2024, 1, 15), bucket_minutes=10
        )
        
        # Should still create buckets for entire day
        assert len(analysis.time_buckets) == 144
        
        # Most buckets should have zero demand
        zero_demand_buckets = [b for b in analysis.time_buckets if b.total_demand == 0]
        assert len(zero_demand_buckets) > 100  # Most of the day should be empty
        
        # Should not have overload windows
        assert len(analysis.overload_windows) == 0
        
        # Peak hour should be identified correctly
        assert analysis.peak_hour in [6, 7]  # One of the hours with flights


if __name__ == "__main__":
    pytest.main([__file__])