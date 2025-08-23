"""Integration test demonstrating complete database workflow."""

import pytest
import tempfile
import shutil
from datetime import datetime, date, time
from pathlib import Path

from src.services.database import FlightDatabaseService, DatabaseConfig
from src.models.flight import Flight, Airport, FlightTime, FlightStatus


def test_complete_database_workflow():
    """Test complete workflow: store flights, query, analyze, export."""
    
    # Setup temporary database
    temp_dir = tempfile.mkdtemp()
    config = DatabaseConfig(
        db_path=str(Path(temp_dir) / "test_workflow.duckdb"),
        parquet_export_path=str(Path(temp_dir) / "parquet")
    )
    
    try:
        # Initialize database service
        db_service = FlightDatabaseService(config)
        
        # Create sample flight data representing a busy day at BOM
        flights = []
        
        # Morning rush (6-9 AM)
        for i in range(10):
            flight = Flight(
                flight_id=f"morning-{i:02d}",
                flight_number=f"AI{2500 + i}",
                airline_code="AI",
                origin=Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai"),
                destination=Airport(code="DEL", name="Delhi (DEL)", city="Delhi"),
                aircraft_type="A320",
                flight_date=date(2024, 1, 15),
                departure=FlightTime(
                    scheduled=time(6 + i // 3, (i % 3) * 20),
                    actual=datetime(2024, 1, 15, 6 + i // 3, (i % 3) * 20 + (i % 5))  # Some delays
                ),
                arrival=FlightTime(
                    scheduled=time(8 + i // 3, (i % 3) * 20),
                    actual=datetime(2024, 1, 15, 8 + i // 3, (i % 3) * 20 + (i % 5))
                ),
                status=FlightStatus.ARRIVED,
                data_source="integration_test"
            )
            flight.dep_delay_min = i % 5  # 0-4 minute delays
            flight.arr_delay_min = i % 5
            flights.append(flight)
        
        # Afternoon flights (12-3 PM)
        for i in range(5):
            flight = Flight(
                flight_id=f"afternoon-{i:02d}",
                flight_number=f"6E{100 + i}",
                airline_code="6E",
                origin=Airport(code="BOM", name="Mumbai (BOM)", city="Mumbai"),
                destination=Airport(code="BLR", name="Bangalore (BLR)", city="Bangalore"),
                aircraft_type="A320",
                flight_date=date(2024, 1, 15),
                departure=FlightTime(
                    scheduled=time(12 + i, 0),
                    actual=datetime(2024, 1, 15, 12 + i, i * 10)  # Increasing delays
                ),
                arrival=FlightTime(
                    scheduled=time(13 + i, 30),
                    actual=datetime(2024, 1, 15, 13 + i, min(59, 30 + i * 10))
                ),
                status=FlightStatus.ARRIVED,
                data_source="integration_test"
            )
            flight.dep_delay_min = i * 10  # 0, 10, 20, 30, 40 minute delays
            flight.arr_delay_min = i * 10
            flights.append(flight)
        
        # Step 1: Store flights in database
        store_result = db_service.store_flights(flights)
        assert store_result["stored"] == 15
        assert store_result["success_rate"] == 1.0
        print(f"✓ Stored {store_result['stored']} flights successfully")
        
        # Step 2: Query flights by date range
        query_result = db_service.query_flights_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15)
        )
        assert query_result.row_count == 15
        print(f"✓ Queried {query_result.row_count} flights for date range")
        
        # Step 3: Analyze peak traffic patterns
        peak_result = db_service.query_peak_traffic(
            airport_code="BOM",
            date_filter=date(2024, 1, 15),
            bucket_minutes=60  # 1-hour buckets
        )
        assert peak_result.row_count > 0
        print(f"✓ Analyzed peak traffic with {peak_result.row_count} time buckets")
        
        # Verify peak analysis shows morning rush
        morning_buckets = [bucket for bucket in peak_result.data 
                          if bucket['time_bucket'].hour >= 6 and bucket['time_bucket'].hour < 9]
        assert len(morning_buckets) > 0
        assert any(bucket['flight_count'] >= 3 for bucket in morning_buckets)
        print("✓ Detected morning rush hour traffic")
        
        # Step 4: Get database statistics
        stats = db_service.get_database_stats()
        assert stats["airline_stats"]["total_flights"] == 15
        assert stats["airport_stats"]["unique_origins"] == 1  # Only BOM
        assert stats["airport_stats"]["unique_destinations"] == 2  # DEL and BLR
        print(f"✓ Database contains {stats['airline_stats']['total_flights']} flights")
        print(f"  - {stats['airport_stats']['unique_routes']} unique routes")
        print(f"  - Data quality: {stats['data_quality']['std_completeness_pct']}% STD completeness")
        
        # Step 5: Export to Parquet for analytics
        export_result = db_service.export_to_parquet()
        assert export_result["success"] is True
        assert export_result["rows_exported"] == 15
        assert Path(export_result["output_path"]).exists()
        print(f"✓ Exported {export_result['rows_exported']} flights to Parquet")
        print(f"  - File size: {export_result['file_size_bytes']} bytes")
        
        # Step 6: Test upsert functionality
        # Update one flight with a longer delay
        updated_flight = flights[0]
        updated_flight.dep_delay_min = 30
        updated_flight.arr_delay_min = 35
        
        upsert_result = db_service.store_flights([updated_flight])
        assert upsert_result["stored"] == 1
        
        # Verify the update
        updated_query = db_service.query_flights_by_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15)
        )
        updated_flight_data = next(
            (f for f in updated_query.data if f['flight_id'] == updated_flight.flight_id), 
            None
        )
        assert updated_flight_data is not None
        assert updated_flight_data['dep_delay_min'] == 30
        print("✓ Successfully updated flight with upsert")
        
        # Step 7: Test query performance with indexes
        start_time = datetime.now()
        for _ in range(10):
            db_service.query_flights_by_date_range(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 15),
                airport_code="BOM"
            )
        query_time = (datetime.now() - start_time).total_seconds()
        assert query_time < 1.0  # Should be fast with indexes
        print(f"✓ Query performance: 10 queries in {query_time:.3f}s")
        
        print("\n🎉 Complete database workflow test passed!")
        print("✅ All DuckDB storage and querying functionality working correctly")
        
    finally:
        # Cleanup
        db_service.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_complete_database_workflow()