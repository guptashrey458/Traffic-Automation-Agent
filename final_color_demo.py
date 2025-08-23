#!/usr/bin/env python3
"""
Final demonstration: Color-coded delay analysis from Excel data
Shows exactly what we can access and how the green/yellow/red system works
"""

from collections import defaultdict
from src.services.data_ingestion import DataIngestionService


def classify_delay_status(delay_minutes):
    """Classify delay into Excel-style color categories"""
    if delay_minutes is None:
        return "⚪ No Data"
    elif delay_minutes <= 15:
        return "🟩 Green (On-time)"
    elif delay_minutes <= 60:
        return "🟨 Yellow (Moderate)"
    else:
        return "🟥 Red (Critical)"


def main():
    print("🚦 FINAL DEMONSTRATION: Excel Color-Coded Delay Analysis")
    print("=" * 65)
    
    # Load Excel data
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    ingestion_service = DataIngestionService()
    result = ingestion_service.ingest_excel_files([excel_file])
    
    flights = result.flights
    print(f"✅ Successfully loaded {len(flights)} flights from Excel")
    
    # === DEMONSTRATE WHAT WE CAN ACCESS ===
    print(f"\n📊 WHAT WE CAN ACCESS FROM YOUR EXCEL FILE:")
    print("-" * 50)
    
    sample = flights[0]
    print(f"✈️  Sample Flight Details:")
    print(f"   Flight: {sample.flight_number} ({sample.airline_code})")
    print(f"   Route: {sample.origin.name} → {sample.destination.name}")
    print(f"   Date: {sample.flight_date}")
    print(f"   Scheduled: {sample.departure.scheduled} → {sample.arrival.scheduled}")
    print(f"   Actual: {sample.departure.actual} → {sample.arrival.actual}")
    print(f"   Departure Delay: {sample.dep_delay_min} minutes")
    print(f"   Arrival Delay: {sample.arr_delay_min} minutes")
    print(f"   Aircraft: {sample.aircraft_type} ({sample.aircraft_registration})")
    print(f"   Status: {sample.status.value}")
    
    # === COLOR-CODED DELAY ANALYSIS ===
    print(f"\n🚦 COLOR-CODED DELAY ANALYSIS (Excel Style)")
    print("-" * 50)
    
    # Classify all flights by delay status
    delay_classification = defaultdict(list)
    
    for flight in flights:
        if flight.dep_delay_min is not None:
            status = classify_delay_status(flight.dep_delay_min)
            delay_classification[status].append(flight)
    
    # Show distribution
    total_with_data = sum(len(flights) for flights in delay_classification.values())
    
    print(f"Delay Status Distribution ({total_with_data} flights):")
    for status in ["🟩 Green (On-time)", "🟨 Yellow (Moderate)", "🟥 Red (Critical)", "⚪ No Data"]:
        count = len(delay_classification[status])
        percentage = (count / total_with_data * 100) if total_with_data > 0 else 0
        print(f"   {status}: {count} flights ({percentage:.1f}%)")
    
    # === DETAILED ANALYSIS BY COLOR ===
    print(f"\n🔍 DETAILED ANALYSIS BY COLOR STATUS")
    print("-" * 45)
    
    # Green flights (excellent performance)
    green_flights = delay_classification["🟩 Green (On-time)"]
    if green_flights:
        print(f"🟩 GREEN FLIGHTS (≤15min delay) - {len(green_flights)} flights:")
        
        # Best airlines
        green_airlines = defaultdict(int)
        for flight in green_flights:
            green_airlines[flight.airline_code] += 1
        
        print("   Top performing airlines:")
        for airline, count in sorted(green_airlines.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {airline}: {count} on-time flights")
        
        # Best routes
        green_routes = defaultdict(int)
        for flight in green_flights:
            green_routes[flight.get_route_key()] += 1
        
        print("   Most reliable routes:")
        for route, count in sorted(green_routes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {route}: {count} on-time flights")
    
    # Yellow flights (moderate delays)
    yellow_flights = delay_classification["🟨 Yellow (Moderate)"]
    if yellow_flights:
        print(f"\n🟨 YELLOW FLIGHTS (16-60min delay) - {len(yellow_flights)} flights:")
        
        delays = [f.dep_delay_min for f in yellow_flights if f.dep_delay_min]
        avg_delay = sum(delays) / len(delays) if delays else 0
        print(f"   Average delay: {avg_delay:.1f} minutes")
        
        # Airlines with most moderate delays
        yellow_airlines = defaultdict(int)
        for flight in yellow_flights:
            yellow_airlines[flight.airline_code] += 1
        
        print("   Airlines with moderate delays:")
        for airline, count in sorted(yellow_airlines.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {airline}: {count} moderate delays")
    
    # Red flights (critical delays)
    red_flights = delay_classification["🟥 Red (Critical)"]
    if red_flights:
        print(f"\n🟥 RED FLIGHTS (>60min delay) - {len(red_flights)} flights:")
        
        delays = [f.dep_delay_min for f in red_flights if f.dep_delay_min]
        avg_delay = sum(delays) / len(delays) if delays else 0
        max_delay = max(delays) if delays else 0
        print(f"   Average delay: {avg_delay:.1f} minutes")
        print(f"   Maximum delay: {max_delay} minutes")
        
        # Worst performers
        worst_flights = sorted(red_flights, key=lambda x: x.dep_delay_min or 0, reverse=True)[:5]
        print("   Worst delays:")
        for i, flight in enumerate(worst_flights, 1):
            print(f"      {i}. {flight.flight_number} ({flight.airline_code}): {flight.dep_delay_min}min")
            print(f"         Route: {flight.get_route_key()}, Date: {flight.flight_date}")
    
    # === HOURLY PATTERNS ===
    print(f"\n🕐 HOURLY DELAY PATTERNS")
    print("-" * 30)
    
    hourly_colors = defaultdict(lambda: defaultdict(int))
    
    for flight in flights:
        if flight.departure.scheduled and flight.dep_delay_min is not None:
            hour = flight.departure.scheduled.hour
            status = classify_delay_status(flight.dep_delay_min)
            hourly_colors[hour][status] += 1
    
    # Show peak hours with color breakdown
    hourly_totals = {hour: sum(colors.values()) for hour, colors in hourly_colors.items()}
    peak_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)[:6]
    
    print("Peak hours with delay status breakdown:")
    for hour, total in peak_hours:
        colors = hourly_colors[hour]
        green = colors.get("🟩 Green (On-time)", 0)
        yellow = colors.get("🟨 Yellow (Moderate)", 0)
        red = colors.get("🟥 Red (Critical)", 0)
        
        green_pct = (green / total * 100) if total > 0 else 0
        yellow_pct = (yellow / total * 100) if total > 0 else 0
        red_pct = (red / total * 100) if total > 0 else 0
        
        print(f"   {hour:02d}:00-{hour:02d}:59 ({total} flights):")
        print(f"      🟩 {green_pct:.1f}% 🟨 {yellow_pct:.1f}% 🟥 {red_pct:.1f}%")
    
    # === EXCEL FORMULAS ===
    print(f"\n📊 EXCEL FORMULAS TO REPLICATE THIS ANALYSIS")
    print("-" * 55)
    
    print("1. Add a 'Delay Status' column with this formula:")
    print('   =IF(K2<=15,"🟩 Green",IF(K2<=60,"🟨 Yellow","🟥 Red"))')
    print("   (Replace K2 with your delay minutes column)")
    
    print(f"\n2. Conditional Formatting Rules:")
    print("   🟩 Green: Cell contains 'Green' → Light green background")
    print("   🟨 Yellow: Cell contains 'Yellow' → Light yellow background") 
    print("   🟥 Red: Cell contains 'Red' → Light red background")
    
    print(f"\n3. Pivot Table Setup:")
    print("   Rows: Airline, Route, Hour")
    print("   Columns: Delay Status")
    print("   Values: Count of Flight Number")
    print("   This gives you the exact same analysis!")
    
    # === OPERATIONAL INSIGHTS ===
    print(f"\n💡 KEY OPERATIONAL INSIGHTS")
    print("-" * 35)
    
    green_count = len(green_flights)
    yellow_count = len(yellow_flights)
    red_count = len(red_flights)
    total = green_count + yellow_count + red_count
    
    if total > 0:
        on_time_rate = green_count / total * 100
        critical_rate = red_count / total * 100
        
        print(f"🎯 Performance Metrics:")
        print(f"   On-time Rate: {on_time_rate:.1f}%")
        print(f"   Critical Delay Rate: {critical_rate:.1f}%")
        
        print(f"\n🚀 What This Enables:")
        print("   ✅ Real-time performance monitoring")
        print("   ✅ Airline benchmarking and comparison")
        print("   ✅ Route optimization opportunities")
        print("   ✅ Peak hour congestion analysis")
        print("   ✅ Predictive delay modeling")
        print("   ✅ Operational decision support")
    
    # === SUMMARY ===
    print(f"\n🎉 SUMMARY: COMPLETE ACCESS TO FLIGHT DATA")
    print("=" * 50)
    
    print(f"✅ {len(flights)} flights loaded from Excel with full details")
    print(f"✅ Color-coded delay classification (Green/Yellow/Red)")
    print(f"✅ Complete route, timing, and aircraft information")
    print(f"✅ Ready for advanced analytics and optimization")
    print(f"✅ Excel formulas provided for manual analysis")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Store in high-performance database (DuckDB)")
    print("2. Build peak traffic analysis algorithms")
    print("3. Create delay prediction models")
    print("4. Develop schedule optimization engine")
    print("5. Build real-time monitoring dashboard")
    
    print(f"\n🎯 YOUR DATA IS READY FOR ADVANCED FLIGHT OPERATIONS ANALYTICS!")


if __name__ == "__main__":
    main()