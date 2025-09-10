#!/usr/bin/env python3
"""
Secure Aire Interactive Demo - Streamlit Web App
Clean version without scoping issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Secure Aire Interactive Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SecureAireGenabilityCalculator:
    def __init__(self, app_id, app_key, customer_name="Customer"):
        self.app_id = app_id
        self.app_key = app_key
        self.auth = HTTPBasicAuth(app_id, app_key)
        self.customer_name = customer_name
        
        # API configuration - from your working script
        self.base_url = "https://api.genability.com/rest/v1"
        self.master_tariff_id = 3154023  # ConEd SC9 - your researched tariff
        
        # Time-of-use factors from your original code
        self.tou_factors = {
            'on_peak_ratio': 0.72,  # 72% of usage during on-peak hours
            'summer_demand_multiplier': 1.15  # 15% higher demand in summer
        }
    
    def load_customer_data(self, df):
        """Load customer data from uploaded DataFrame"""
        # Flexible column detection (from your script)
        timestamp_col = None
        power_col = None
        
        # Look for timestamp column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date']):
                timestamp_col = col
                break
        
        # Look for power column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['electric', 'power', 'kw', 'demand']):
                if 'total' in col.lower() or 'electric' in col.lower():
                    power_col = col
                    break
        
        if not timestamp_col:
            timestamp_col = df.columns[0]
            st.warning(f"Using first column as timestamp: {timestamp_col}")
        
        if not power_col:
            power_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            st.warning(f"Using column as power: {power_col}")
        
        # Rename columns for consistency
        df = df.rename(columns={
            timestamp_col: 'Timestamp',
            power_col: 'Total Electric (kW)'
        })
        
        # Ensure timestamp is datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Remove any rows with missing data
        df = df.dropna(subset=['Timestamp', 'Total Electric (kW)'])
        
        st.success(f"✅ Loaded {len(df)} data points from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        return df
    
    def determine_tariff_type(self, peak_demand_kw):
        """Determine if customer should use Rate I or Rate II based on demand"""
        if peak_demand_kw >= 500:
            return 'SC9 Rate II (Large TOD)', True
        else:
            return 'SC9 Rate I (Standard)', False
    
    def estimate_reactive_power(self, peak_kw, building_age_factor=1.0):
        """Estimate reactive power demand - from your original code"""
        power_factor = min(0.95, 0.85 + (0.10 * building_age_factor))
        kva = peak_kw / power_factor
        kvar = np.sqrt(kva**2 - peak_kw**2)
        billable_kvar = max(0, kvar - (peak_kw / 3))
        return round(billable_kvar)
    
    def process_monthly_data(self, df, year_month, reduction_factor=0):
        """Process data for a specific month with reduction applied to ALL intervals"""
        month_data = df[df['Timestamp'].dt.to_period('M') == year_month].copy()
        
        if len(month_data) == 0:
            return None
        
        # Apply reduction to ALL intervals (not just peaks)
        if reduction_factor > 0:
            month_data['Total Electric (kW)'] = month_data['Total Electric (kW)'] * (1 - reduction_factor)
        
        # Calculate energy consumption (kWh) from power (kW) readings
        time_diff = (month_data['Timestamp'].iloc[1] - month_data['Timestamp'].iloc[0]).total_seconds() / 60
        hours_per_interval = time_diff / 60
        month_data['kWh_interval'] = month_data['Total Electric (kW)'] * hours_per_interval
        
        # Time-of-use classification
        month_data['hour'] = month_data['Timestamp'].dt.hour
        month_data['weekday'] = month_data['Timestamp'].dt.weekday
        month_data['month_num'] = month_data['Timestamp'].dt.month
        
        # NYC ConEd on-peak hours: 8 AM to 10 PM weekdays
        month_data['tou_period'] = 'off_peak'
        on_peak_mask = (
            (month_data['hour'] >= 8) & 
            (month_data['hour'] < 22) &
            (month_data['weekday'] < 5)
        )
        month_data.loc[on_peak_mask, 'tou_period'] = 'on_peak'
        
        # Calculate monthly summaries
        total_kwh = month_data['kWh_interval'].sum()
        on_peak_kwh = month_data[month_data['tou_period'] == 'on_peak']['kWh_interval'].sum()
        off_peak_kwh = total_kwh - on_peak_kwh
        peak_demand_kw = month_data['Total Electric (kW)'].max()
        
        # Find the exact timestamp of the peak demand
        peak_idx = month_data['Total Electric (kW)'].idxmax()
        peak_timestamp = month_data.loc[peak_idx, 'Timestamp']
        peak_date = peak_timestamp.strftime('%Y-%m-%d')
        peak_time = peak_timestamp.strftime('%H:%M')
        peak_day_of_week = peak_timestamp.strftime('%A')
        
        # Calculate demand by TOU period
        on_peak_data = month_data[month_data['tou_period'] == 'on_peak']
        off_peak_data = month_data[month_data['tou_period'] == 'off_peak']
        
        monthly_summary = {
            'year_month': year_month,
            'total_kwh': total_kwh,
            'on_peak_kwh': on_peak_kwh,
            'off_peak_kwh': off_peak_kwh,
            'peak_demand_kw': peak_demand_kw,
            'peak_timestamp': peak_timestamp,  # Add actual peak timestamp
            'peak_date': peak_date,           # Add formatted date
            'peak_time': peak_time,           # Add formatted time  
            'peak_day_of_week': peak_day_of_week,  # Add day of week
            'on_peak_demand_kw': on_peak_data['Total Electric (kW)'].max() if len(on_peak_data) > 0 else 0,
            'off_peak_demand_kw': off_peak_data['Total Electric (kW)'].max() if len(off_peak_data) > 0 else 0,
            'avg_demand_kw': month_data['Total Electric (kW)'].mean(),
            'load_factor': month_data['Total Electric (kW)'].mean() / peak_demand_kw if peak_demand_kw > 0 else 0,
            'data_points': len(month_data),
            'interval_minutes': time_diff,
            'reduction_applied': reduction_factor
        }
        
        return monthly_summary
    
    def calculate_costs_genability(self, monthly_summary, year=2024):
        """Calculate costs using Genability API - from your working script"""
        year_month = monthly_summary['year_month']
        month_num = int(str(year_month).split('-')[1])
        
        # Determine tariff type based on peak demand
        tariff_name, is_rate_ii = self.determine_tariff_type(monthly_summary['peak_demand_kw'])
        
        # Build date range
        start_date = f"{year}-{month_num:02d}-01"
        if month_num == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month_num+1:02d}-01"
        
        # Build property inputs (from your working script)
        property_inputs = [
            {
                "keyName": "consumption",
                "dataValue": str(int(monthly_summary['on_peak_kwh'])),
                "period": "ON_PEAK",
                "quantityUnit": "kWh"
            },
            {
                "keyName": "consumption",
                "dataValue": str(int(monthly_summary['off_peak_kwh'])),
                "period": "OFF_PEAK",
                "quantityUnit": "kWh"
            }
        ]
        
        # Add demand inputs for Rate II
        peak_demand = int(monthly_summary['peak_demand_kw'])
        if is_rate_ii:
            property_inputs.extend([
                {
                    "keyName": "billingDemand691",
                    "dataValue": str(peak_demand),
                    "quantityUnit": "kW"
                },
                {
                    "keyName": "billingDemand691",
                    "dataValue": str(peak_demand),
                    "period": "ON_PEAK",
                    "quantityUnit": "kW"
                },
                {
                    "keyName": "billingDemand691",
                    "dataValue": str(int(monthly_summary['off_peak_demand_kw'])),
                    "period": "OFF_PEAK",
                    "quantityUnit": "kW"
                }
            ])
            
            # Add reactive power for large buildings
            reactive_kvar = self.estimate_reactive_power(peak_demand)
            if reactive_kvar > 0:
                property_inputs.append({
                    "keyName": "reactiveDemand691",
                    "dataValue": str(reactive_kvar),
                    "quantityUnit": "kVAR"
                })
        
        # Build API payload (exactly from your script)
        payload = {
            "fromDateTime": f"{start_date}T00:00:00-05:00",
            "toDateTime": f"{end_date}T00:00:00-05:00",
            "masterTariffId": self.master_tariff_id,
            "groupBy": "ALL",
            "detailLevel": "CHARGE_TYPE_AND_TOU",
            "billingPeriod": True,
            "propertyInputs": property_inputs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/ondemand/calculate",
                json=payload,
                auth=self.auth,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success' and result['results']:
                    calc = result['results'][0]
                    
                    summary = {
                        'year_month': year_month,
                        'tariff': tariff_name,
                        'kwh_total': monthly_summary['total_kwh'],
                        'kwh_on_peak': monthly_summary['on_peak_kwh'],
                        'kwh_off_peak': monthly_summary['off_peak_kwh'],
                        'kw_peak': calc.get('summary', {}).get('kW', peak_demand),
                        'kvar': calc.get('summary', {}).get('kVAR', 0),
                        'total_cost': calc.get('totalCost', 0),
                        'load_factor': monthly_summary['load_factor'],
                        'reduction_applied': monthly_summary.get('reduction_applied', 0),
                        'api_details': calc.get('items', [])  # Save API details for breakdown
                    }
                    
                    # Extract detailed charges
                    energy_charges = sum(item.get('cost', 0) for item in calc.get('items', []) if 'energy' in item.get('chargeType', '').lower())
                    demand_charges = sum(item.get('cost', 0) for item in calc.get('items', []) if 'demand' in item.get('chargeType', '').lower())
                    
                    summary['energy_charges'] = energy_charges
                    summary['demand_charges'] = demand_charges
                    
                    return summary
                else:
                    st.error(f"API returned error: {result.get('status', 'unknown')}")
                    return None
            else:
                st.error(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error calling Genability API: {str(e)}")
            return None
    
    def process_all_months(self, df, reduction_factor=0):
        """Process all months with the given reduction factor"""
        # Get all unique months in the data
        months = df['Timestamp'].dt.to_period('M').unique()
        months = sorted(months)
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, month in enumerate(months):
            status_text.text(f"Processing {month} ({i+1}/{len(months)})...")
            progress_bar.progress((i + 1) / len(months))
            
            monthly_summary = self.process_monthly_data(df, month, reduction_factor)
            
            if monthly_summary and monthly_summary['total_kwh'] > 0:
                cost_summary = self.calculate_costs_genability(monthly_summary)
                
                if cost_summary:
                    results[str(month)] = {**monthly_summary, **cost_summary}
                    
                    # Rate limiting
                    time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        return results

# Streamlit App Layout
def main():
    st.title("🚀 Secure Aire Interactive Demo")
    st.markdown("### Real Genability API Cost Analysis - Upload Your 15-Minute Interval Data")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API credentials - user must enter their own
    app_id = st.sidebar.text_input("Genability App ID", value="", help="Enter your Genability App ID")
    app_key = st.sidebar.text_input("Genability App Key", value="", type="password", help="Enter your Genability App Key")
    
    # Customer name
    customer_name = st.sidebar.text_input("Customer Name", value="Customer")
    
    # Validate credentials are entered
    if not app_id or not app_key:
        st.warning("⚠️ Please enter your Genability API credentials in the sidebar to use this app.")
        st.info("💡 You can get API credentials from your Genability account at https://genability.com")
        return  # Exit early if no credentials
    
    # MOVED PEAK REDUCTION SLIDER TO TOP
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        reduction_percent = st.slider(
            "🎯 Peak Reduction Percentage",
            min_value=1,
            max_value=25,
            value=10,
            help="Percentage reduction applied to ALL intervals in each month"
        )
    
    st.markdown(f"### Set to {reduction_percent}% reduction - Upload your data below to calculate savings")
    
    # Instructions - COLLAPSED BY DEFAULT
    with st.expander("📋 How This Demo Works", expanded=False):
        st.markdown("""
        1. **Upload** your 15-minute interval CSV data (must have 'Timestamp' and 'Total Electric (kW)' columns)
        2. **System finds** monthly peak intervals and runs entire months through Genability API  
        3. **Adjust slider** to set reduction percentage (default 10%)
        4. **View real-time** before/after cost comparison with actual API rates
        5. **Perfect for** client presentations - shows real dollar savings!
        
        **Key Insight:** Reduction is applied to ALL intervals in each month, not just peaks!
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload 15-Minute Interval CSV", 
        type=['csv'],
        help="CSV must contain timestamp and power consumption columns"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show data preview
            with st.expander("🔍 Data Preview", expanded=False):
                st.write("**Columns found:**", list(df.columns))
                st.dataframe(df.head(10))
                st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Initialize calculator
            calculator = SecureAireGenabilityCalculator(app_id, app_key, customer_name)
            
            # Process data
            processed_df = calculator.load_customer_data(df)
            
            if processed_df is not None and len(processed_df) > 0:
                # Single process button - no need for slider here anymore since it's at the top
                if st.button("🔄 Calculate Costs with Genability API", type="primary", use_container_width=True):
                    
                    # Calculate baseline costs
                    st.markdown("#### 📊 Calculating Baseline Costs...")
                    baseline_results = calculator.process_all_months(processed_df, reduction_factor=0)
                    
                    if baseline_results:
                        # Calculate reduced costs
                        st.markdown("#### 🎯 Calculating Costs with Secure Aire Reduction...")
                        reduced_results = calculator.process_all_months(processed_df, reduction_factor=reduction_percent/100)
                        
                        if reduced_results:
                            # Display results
                            try:
                                display_results(baseline_results, reduced_results, reduction_percent)
                            except Exception as e:
                                st.error(f"❌ Error displaying results: {str(e)}")
                        else:
                            st.error("❌ Failed to calculate reduced costs")
                    else:
                        st.error("❌ Failed to calculate baseline costs")
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

def display_results(baseline_results, reduced_results, reduction_percent):
    """Display the comparison results with charts - CLEAN VERSION"""
    
    # Calculate totals
    baseline_total = sum(month['total_cost'] for month in baseline_results.values())
    reduced_total = sum(month['total_cost'] for month in reduced_results.values())
    total_savings = baseline_total - reduced_total
    savings_percent = (total_savings / baseline_total) * 100
    
    st.markdown("---")
    st.markdown("## 🎉 Results - Real Genability API Costs")
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Annual Cost",
            f"${baseline_total:,.0f}",
            help="Based on Genability API"
        )
    
    with col2:
        st.metric(
            "With Secure Aire",
            f"${reduced_total:,.0f}",
            delta=f"-${total_savings:,.0f}",
            delta_color="inverse",
            help=f"{reduction_percent}% reduction applied"
        )
    
    with col3:
        st.metric(
            "Annual Savings",
            f"${total_savings:,.0f}",
            delta=f"{savings_percent:.1f}% saved",
            help="Total annual cost reduction"
        )
    
    with col4:
        st.metric(
            "Monthly Average",
            f"${total_savings/12:,.0f}",
            help="Average monthly savings"
        )
    
    # Simple charts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Get data for charts
    months_list = list(baseline_results.keys())
    baseline_costs = [baseline_results[m]['total_cost'] for m in months_list]
    reduced_costs = [reduced_results[m]['total_cost'] for m in months_list]
    baseline_peaks = [baseline_results[m]['kw_peak'] for m in months_list]
    reduced_peaks = [reduced_results[m]['kw_peak'] for m in months_list]
    
    with col1:
        # Monthly comparison chart
        st.markdown("#### 📊 Monthly Cost Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current Cost',
            x=months_list,
            y=baseline_costs,
            marker_color='rgb(220, 53, 69)',
            text=[f'${cost:,.0f}' for cost in baseline_costs],
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            name='With Secure Aire',
            x=months_list,
            y=reduced_costs,
            marker_color='rgb(40, 167, 69)',
            text=[f'${cost:,.0f}' for cost in reduced_costs],
            textposition='auto'
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced peak demand cost chart
        st.markdown("#### 🔥 Peak Demand Costs: Your Most Expensive 15-Minute Periods")
        
        # Calculate demand costs for each month
        baseline_demand_costs = [baseline_results[m].get('demand_charges', 0) for m in months_list]
        reduced_demand_costs = [reduced_results[m].get('demand_charges', 0) for m in months_list]
        
        fig = go.Figure()
        
        # Baseline demand costs (red bars)
        fig.add_trace(go.Bar(
            name='Current Demand Cost',
            x=months_list,
            y=baseline_demand_costs,
            marker_color='rgb(220, 53, 69)',
            text=[f'{peak:,.0f} kW<br>${cost/1000:.0f}K' for peak, cost in zip(baseline_peaks, baseline_demand_costs)],
            textposition='auto',
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{x}</b><br>' +
                         'Peak: %{customdata[0]} %{customdata[1]}<br>' +
                         'Demand: %{customdata[2]:,.0f} kW<br>' +
                         'Cost: $%{y:,.0f}<br>' +
                         '<extra></extra>',
            customdata=[[baseline_results[m].get('peak_date', 'Unknown'), 
                        baseline_results[m].get('peak_time', 'Unknown'),
                        baseline_results[m]['kw_peak']] for m in months_list],
            yaxis='y'
        ))
        
        # Reduced demand costs (green bars)
        fig.add_trace(go.Bar(
            name='Reduced Demand Cost', 
            x=months_list,
            y=reduced_demand_costs,
            marker_color='rgb(40, 167, 69)',
            text=[f'{peak:,.0f} kW<br>${cost/1000:.0f}K' for peak, cost in zip(reduced_peaks, reduced_demand_costs)],
            textposition='auto',
            textfont=dict(color='white', size=10),
            hovertemplate='<b>%{x}</b><br>' +
                         'Peak: %{customdata[0]} %{customdata[1]}<br>' +
                         'Demand: %{customdata[2]:,.0f} kW<br>' +
                         'Cost: $%{y:,.0f}<br>' +
                         'Reduction Applied<br>' +
                         '<extra></extra>',
            customdata=[[reduced_results[m].get('peak_date', 'Unknown'), 
                        reduced_results[m].get('peak_time', 'Unknown'),
                        reduced_results[m]['kw_peak']] for m in months_list],
            yaxis='y'
        ))
        
        # Add line showing cost per minute of peak (secondary y-axis)
        cost_per_minute = [cost / 15 for cost in baseline_demand_costs]  # 15 minutes per peak period
        
        fig.add_trace(go.Scatter(
            name='Cost per Peak Minute',
            x=months_list,
            y=cost_per_minute,
            mode='lines+markers',
            line=dict(color='rgb(255, 99, 132)', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            text=[f'${cost:,.0f}/min' for cost in cost_per_minute],
            textposition='top center',
            yaxis='y2'
        ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            barmode='group',
            xaxis_title="Month", 
            yaxis=dict(
                title="Monthly Demand Charges ($)",
                side='left'
            ),
            yaxis2=dict(
                title="Cost per Peak Minute ($/min)",
                side='right',
                overlaying='y'
            ),
            template="plotly_white",
            height=400,
            legend=dict(x=0.01, y=0.99)
        )
        
        # Add annotation for highest cost month with actual peak date/time
        if baseline_demand_costs:
            max_cost_idx = baseline_demand_costs.index(max(baseline_demand_costs))
            max_cost_month = months_list[max_cost_idx]
            max_cost = baseline_demand_costs[max_cost_idx]
            max_peak = baseline_peaks[max_cost_idx]
            
            # Get the actual peak date/time from baseline results
            peak_info = baseline_results[max_cost_month]
            peak_date = peak_info.get('peak_date', 'Unknown')
            peak_time = peak_info.get('peak_time', 'Unknown')
            peak_day = peak_info.get('peak_day_of_week', 'Unknown')
            
            fig.add_annotation(
                x=max_cost_month,
                y=max_cost,
                text=f"Peak: {peak_date}<br>{peak_day} {peak_time}<br>{max_peak:,.0f} kW<br>${max_cost/1000:.0f}K/month<br>${max_cost/15:,.0f}/minute",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="white",
                bordercolor="red",
                font=dict(size=9)
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("📈 Detailed Monthly Breakdown", expanded=False):
        breakdown_data = []
        
        for month in months_list:
            baseline = baseline_results[month]
            reduced = reduced_results[month]
            savings = baseline['total_cost'] - reduced['total_cost']
            
            breakdown_data.append({
                'Month': month,
                'Baseline Cost': f"${baseline['total_cost']:,.0f}",
                'Reduced Cost': f"${reduced['total_cost']:,.0f}",
                'Savings': f"${savings:,.0f}",
                'Peak kW (Baseline)': f"{baseline['kw_peak']:,.0f}",
                'Peak kW (Reduced)': f"{reduced['kw_peak']:,.0f}",
                'Energy (kWh)': f"{baseline['kwh_total']:,.0f}",
                'Tariff': baseline.get('tariff', 'SC9')
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)
    
    # Export results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Download Results as CSV", use_container_width=True):
            # Create export data
            export_data = []
            for month in months_list:
                baseline = baseline_results[month]
                reduced = reduced_results[month]
                export_data.append({
                    'month': month,
                    'baseline_cost': baseline['total_cost'],
                    'reduced_cost': reduced['total_cost'],
                    'savings': baseline['total_cost'] - reduced['total_cost'],
                    'baseline_peak_kw': baseline['kw_peak'],
                    'reduced_peak_kw': reduced['kw_peak'],
                    'energy_kwh': baseline['kwh_total'],
                    'reduction_percent': reduction_percent
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"secure_aire_results_{reduction_percent}pct.csv",
                mime="text/csv"
            )
    
    with col2:
        st.success(f"✅ Analysis complete! Real API costs calculated for {len(months_list)} months")

if __name__ == "__main__":
    main()