#!/usr/bin/env python3
"""
Enhanced delay analysis with color-coded status indicators
Implementing the green/yellow/red delay classification system
"""

from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import pandas as pd

from src.services.data_ingestion import DataIngestionService


class DelayStatus(Enum):
    """Delay status based on color coding system"""
    ON_TIME = "green"      # ≤ 15 minutes
    MODERATE = "yellow"    # 16-60 minutes  
    CRITICAL = "red"       # > 60 minutes


def classify_delay_status(delay_minutes):
    """
    Classify delay into green/yellow/red categories
    Based on common aviation industry standards
    """
    if delay_minutes is None:
        return None
    
    if delay_minutes <= 15:
        return DelayStatus.ON_TIME
    elif delay_minutes <= 60:
        return DelayStatus.MODERATE
    else:
        return DelayStatus.CRITICAL


def analyze_delay_patterns_with_status():
    """Comprehensive delay analysis with color-coded status"""
    
    print("🚦 ENHANCED DELAY ANALYSIS WITH STATUS INDICATORS")
    print("=" * 60)
    
    # Load flight data
    excel_file = "429e6e3f-281d-4e4c-b00a-92fb020cb2fcFlight_Data.xlsx"
    ingestion_service = DataIngestionService()
    result = ingestion_service.ingest_excel_files([excel_file])
    
    flights = result.flights
    print(f"📊 Analyzing {len(flights)} flights with delay status classification")
    
    # === DELAY STATUS DISTRIBUTION ===
    print(f"\n🚦 DELAY STATUS DISTRIBUTION")
    print("-" * 35)
    
    status_counts = {
        DelayStatus.ON_TIME: 0,
        DelayStatus.MODERATE: 0, 
        DelayStatus.CRITICAL: 0,
        None: 0
    }
    
    departure_delays = []
    arrival_delays = []
    
    for flight in flights:
        # Departure delay status
        if flight.dep_delay_min is not None:
            dep_status = classify_delay_status(flight.dep_delay_min)
            status_counts[dep_status] += 1
            departure_delays.append((flight, dep_status, flight.dep_delay_min))
        else:
            status_counts[None] += 1
    
    total_with_data = len(flights) - status_counts[None]
    
    print(f"🟩 GREEN (On-time ≤15min): {status_counts[DelayStatus.ON_TIME]} flights ({status_counts[DelayStatus.ON_TIME]/total_with_data*100:.1f}%)")
    print(f"🟨 YELLOW (Moderate 16-60min): {status_counts[DelayStatus.MODERATE]} flights ({status_counts[DelayStatus.MODERATE]/total_with_data*100:.1f}%)")
    print(f"🟥 RED (Critical >60min): {status_counts[DelayStatus.CRITICAL]} flights ({status_counts[DelayStatus.CRITICAL]/total_with_data*100:.1f}%)")
    print(f"⚪ No data: {status_counts[None]} flights")
    
    # === AIRLINE PERFORMANCE BY STATUS ===
    print(f"\n🏢 AIRLINE PERFORMANCE BY DELAY STATUS")
    print("-" * 45)
    
    airline_status = defaultdict(lambda: {
        DelayStatus.ON_TIME: 0,
        DelayStatus.MODERATE: 0,
        DelayStatus.CRITICAL: 0
    })
    
    for flight, status, delay in departure_delays:
        if status:
            airline_status[flight.airline_code][status] += 1
    
    print("Airline performance (Green/Yellow/Red):")
    for airline in sorted(airline_status.keys()):
        stats = airline_status[airline]
        total = sum(stats.values())
        
        green_pct = stats[DelayStatus.ON_TIME] / total * 100
        yellow_pct = stats[DelayStatus.MODERATE] / total * 100
        red_pct = stats[DelayStatus.CRITICAL] / total * 100
        
        print(f"   {airline}: 🟩{green_pct:.1f}% 🟨{yellow_pct:.1f}% 🟥{red_pct:.1f}% ({total} flights)")
    
    # === ROUTE PERFORMANCE BY STATUS ===
    print(f"\n🗺️  ROUTE PERFORMANCE BY DELAY STATUS")
    print("-" * 40)
    
    route_status = defaultdict(lambda: {
        DelayStatus.ON_TIME: 0,
        DelayStatus.MODERATE: 0,
        DelayStatus.CRITICAL: 0
    })
    
    for flight, status, delay in departure_delays:
        if status:
            route = flight.get_route_key()
            route_status[route][status] += 1
    
    # Show top 10 busiest routes
    route_totals = {route: sum(stats.values()) for route, stats in route_status.items()}
    top_routes = sorted(route_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top routes performance (Green/Yellow/Red):")
    for route, total in top_routes:
        stats = route_status[route]
        
        green_pct = stats[DelayStatus.ON_TIME] / total * 100
        yellow_pct = stats[DelayStatus.MODERATE] / total * 100
        red_pct = stats[DelayStatus.CRITICAL] / total * 100
        
        print(f"   {route}: 🟩{green_pct:.1f}% 🟨{yellow_pct:.1f}% 🟥{red_pct:.1f}% ({total} flights)")
    
    # === TIME-BASED DELAY PATTERNS ===
    print(f"\n🕐 HOURLY DELAY STATUS PATTERNS")
    print("-" * 35)
    
    hourly_status = defaultdict(lambda: {
        DelayStatus.ON_TIME: 0,
        DelayStatus.MODERATE: 0,
        DelayStatus.CRITICAL: 0
    })
    
    for flight, status, delay in departure_delays:
        if status and flight.departure.scheduled:
            hour = flight.departure.scheduled.hour
            hourly_status[hour][status] += 1
    
    print("Peak hours delay status distribution:")
    # Sort by total flights per hour
    hourly_totals = {hour: sum(stats.values()) for hour, stats in hourly_status.items()}
    peak_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)[:8]
    
    for hour, total in peak_hours:
        stats = hourly_status[hour]
        
        green_pct = stats[DelayStatus.ON_TIME] / total * 100
        yellow_pct = stats[DelayStatus.MODERATE] / total * 100
        red_pct = stats[DelayStatus.CRITICAL] / total * 100
        
        print(f"   {hour:02d}:00-{hour:02d}:59: 🟩{green_pct:.1f}% 🟨{yellow_pct:.1f}% 🟥{red_pct:.1f}% ({total} flights)")
    
    # === WORST PERFORMERS ANALYSIS ===
    print(f"\n🚨 CRITICAL DELAY ANALYSIS (RED STATUS)")
    print("-" * 40)
    
    critical_delays = [(flight, delay) for flight, status, delay in departure_delays 
                      if status == DelayStatus.CRITICAL]
    
    if critical_delays:
        print(f"Total critical delays (>60min): {len(critical_delays)}")
        
        # Worst delays
        worst_delays = sorted(critical_delays, key=lambda x: x[1], reverse=True)[:5]
        print(f"\nWorst delays:")
        for i, (flight, delay) in enumerate(worst_delays, 1):
            print(f"   {i}. {flight.flight_number} ({flight.airline_code}): {delay} minutes")
            print(f"      Route: {flight.get_route_key()}")
            print(f"      Date: {flight.flight_date}")
            print(f"      Aircraft: {flight.aircraft_type}")
        
        # Critical delay patterns
        critical_airlines = defaultdict(int)
        critical_routes = defaultdict(int)
        critical_hours = defaultdict(int)
        
        for flight, delay in critical_delays:
            critical_airlines[flight.airline_code] += 1
            critical_routes[flight.get_route_key()] += 1
            if flight.departure.scheduled:
                critical_hours[flight.departure.scheduled.hour] += 1
        
        print(f"\nCritical delay patterns:")
        print(f"   Airlines with most critical delays:")
        for airline, count in sorted(critical_airlines.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {airline}: {count} critical delays")
        
        print(f"   Routes with most critical delays:")
        for route, count in sorted(critical_routes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {route}: {count} critical delays")
        
        print(f"   Hours with most critical delays:")
        for hour, count in sorted(critical_hours.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {hour:02d}:00-{hour:02d}:59: {count} critical delays")
    
    # === EXCEL CONDITIONAL FORMATTING RULES ===
    print(f"\n📊 EXCEL CONDITIONAL FORMATTING RULES")
    print("-" * 45)
    
    print("To replicate the color coding in Excel:")
    print("1. Select the delay columns (e.g., column with delay minutes)")
    print("2. Go to Home → Conditional Formatting → New Rule")
    print("3. Use these formulas:")
    print()
    print("🟩 GREEN (On-time): Cell Value ≤ 15")
    print("   Formula: =$K2<=15")
    print("   Format: Green fill, dark green text")
    print()
    print("🟨 YELLOW (Moderate): Cell Value > 15 AND ≤ 60") 
    print("   Formula: =AND($K2>15,$K2<=60)")
    print("   Format: Yellow fill, dark yellow text")
    print()
    print("🟥 RED (Critical): Cell Value > 60")
    print("   Formula: =$K2>60") 
    print("   Format: Red fill, white text")
    
    # === RECOMMENDATIONS ===
    print(f"\n💡 OPERATIONAL RECOMMENDATIONS")
    print("-" * 35)
    
    print("Based on delay status analysis:")
    print()
    print("🟩 Maintain Green Performance:")
    print("   - Focus on routes with >70% on-time rate")
    print("   - Replicate best practices from top performers")
    print()
    print("🟨 Improve Yellow Performance:")
    print("   - Target moderate delays for quick wins")
    print("   - Implement buffer time in schedules")
    print()
    print("🟥 Address Red Critical Issues:")
    print("   - Investigate root causes of >60min delays")
    print("   - Consider schedule adjustments for problem routes")
    print("   - Focus on peak hour congestion management")
    
    return flights, departure_delays


def generate_delay_status_report(flights, delays):
    """Generate a comprehensive delay status report"""
    
    print(f"\n📋 DELAY STATUS SUMMARY REPORT")
    print("=" * 40)
    
    total_flights = len([d for d in delays if d[1] is not None])
    
    green_count = len([d for d in delays if d[1] == DelayStatus.ON_TIME])
    yellow_count = len([d for d in delays if d[1] == DelayStatus.MODERATE])
    red_count = len([d for d in delays if d[1] == DelayStatus.CRITICAL])
    
    print(f"Overall Performance Score:")
    print(f"🟩 On-time Rate: {green_count/total_flights*100:.1f}%")
    print(f"🟨 Moderate Delay Rate: {yellow_count/total_flights*100:.1f}%")
    print(f"🟥 Critical Delay Rate: {red_count/total_flights*100:.1f}%")
    
    # Performance grade
    on_time_rate = green_count/total_flights*100
    if on_time_rate >= 80:
        grade = "A+ (Excellent)"
    elif on_time_rate >= 70:
        grade = "A (Good)"
    elif on_time_rate >= 60:
        grade = "B (Average)"
    elif on_time_rate >= 50:
        grade = "C (Below Average)"
    else:
        grade = "D (Poor)"
    
    print(f"\n🎯 Performance Grade: {grade}")
    
    return {
        'total_flights': total_flights,
        'green_count': green_count,
        'yellow_count': yellow_count,
        'red_count': red_count,
        'on_time_rate': on_time_rate,
        'grade': grade
    }


if __name__ == "__main__":
    flights, delays = analyze_delay_patterns_with_status()
    report = generate_delay_status_report(flights, delays)
    
    print(f"\n🎉 ENHANCED DELAY ANALYSIS COMPLETE!")
    print(f"✅ Color-coded delay classification implemented")
    print(f"✅ {report['total_flights']} flights analyzed")
    print(f"✅ Performance grade: {report['grade']}")