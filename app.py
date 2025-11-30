# volhedge_dashboard.py
# STREAMLIT DASHBOARD: VolHedge Live Monitor
# PRODUCTION VERSION - FIXED TIMESTAMP HANDLING

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="VolHedge Live", layout="wide", page_icon="📊")

LOG_FILE = 'volhedge_gp_live_log.csv'
CSV_PATH = 'btc_iv_history.csv'

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_mm_ss_timestamp(time_str):
    """
    Parse MM:SS.f format to sequential index
    Handles timestamp resets by maintaining continuity
    """
    try:
        parts = str(time_str).split(':')
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds  # Total seconds
    except:
        return np.nan

def load_log():
    """Load and process log file with robust timestamp handling"""
    try:
        if not os.path.exists(LOG_FILE):
            return pd.DataFrame(), "File not found"
        
        # Read CSV
        df = pd.read_csv(LOG_FILE)
        
        if df.empty:
            return pd.DataFrame(), "File is empty"
        
        # Convert MM:SS.f format to sequential seconds
        df['time_seconds'] = df['timestamp'].apply(parse_mm_ss_timestamp)
        
        # Create sequential index for plotting (row number)
        df['seq_index'] = range(len(df))
        
        # For display purposes, keep original timestamp
        df['timestamp_display'] = df['timestamp']
        
        # Validate numeric columns
        numeric_cols = ['btc_price', 'sm_delta', 'slope_mean', 'slope_std', 'hedge_ratio', 'pnl']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def load_history():
    """Load historical volatility data"""
    try:
        if not os.path.exists(CSV_PATH):
            return pd.DataFrame()
        
        df = pd.read_csv(CSV_PATH, parse_dates=['Dates'])
        return df.tail(30)
        
    except:
        return pd.DataFrame()

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title("📊 VolHedge: On-Chain Delta Hedging with GP Uncertainty")
st.markdown("*Live on Sepolia* | Chainlink Oracle | Gaussian Process Smile | 95% CI Delta")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 60, 10)
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Display settings
show_diagnostics = st.sidebar.checkbox("Show Diagnostics", value=False)
max_points = st.sidebar.slider("Max data points to display", 10, 100, 50, 10)

# Load data
log_df, error = load_log()
hist_df = load_history()

# Show diagnostics if enabled
if show_diagnostics:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Diagnostics")
    st.sidebar.text(f"Working Dir:\n{os.getcwd()}")
    st.sidebar.text(f"Log exists: {os.path.exists(LOG_FILE)}")
    st.sidebar.text(f"Records: {len(log_df)}")

# Check for errors
if error:
    st.error(f"❌ {error}")
    st.info("**Troubleshooting:**\n- Ensure `volhedge_gp_live.py` is running\n- Check file path and permissions")
    st.stop()

if log_df.empty:
    st.warning("⏳ No data available yet. Waiting for first bot execution...")
    st.stop()

# Limit data points for performance
display_df = log_df.tail(max_points).copy()

# ==============================================================================
# KPI METRICS
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)
latest = display_df.iloc[-1]
prev_pnl = display_df['pnl'].iloc[-2] if len(display_df) > 1 else 0

with col1:
    st.metric(
        "On-Chain PnL", 
        f"{latest['pnl']:+.2f} HEDGE", 
        delta=f"{latest['pnl']}",
        delta_color="normal"
    )

with col2:
    st.metric(
        "SM Delta (95% CI)", 
        f"{latest['sm_delta']:.4f}",
        delta=f"{latest['sm_delta'] - display_df['sm_delta'].iloc[-2]:.4f}" if len(display_df) > 1 else None
    )

with col3:
    st.metric(
        "Hedge Ratio", 
        f"{int(latest['hedge_ratio'])}",
        delta=f"{int(latest['hedge_ratio'] - display_df['hedge_ratio'].iloc[-2])}" if len(display_df) > 1 else None
    )

with col4:
    st.metric(
        "ETH Oracle Price", 
        f"${latest['btc_price']:,.2f}",
        delta=f"${latest['btc_price'] - display_df['btc_price'].iloc[-2]:+.2f}" if len(display_df) > 1 else None
    )

# ==============================================================================
# VOLATILITY METRICS
# ==============================================================================
col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Current Volatility", f"{latest['volatility']:.4f}")

with col6:
    st.metric("Slope Mean", f"{latest['slope_mean']:.6f}")

with col7:
    st.metric("Slope Std Dev", f"{latest['slope_std']:.6f}")

st.markdown("---")

# ==============================================================================
# INTERACTIVE CHARTS
# ==============================================================================

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "PnL Evolution", "Hedge Ratio Adjustments",
        "SM Delta (95% CI)", "Volatility Smile Slope",
        "ETH Price Oracle", "Volatility Surface"
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}]
    ]
)

# Use sequential index for x-axis to avoid timestamp discontinuities
x_vals = display_df['seq_index']
x_labels = display_df['timestamp_display']

# 1. PnL Evolution
fig.add_trace(
    go.Scatter(
        x=x_vals, 
        y=display_df['pnl'],
        mode='lines+markers',
        name='PnL',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=6),
        hovertemplate='Time: %{text}<br>PnL: %{y:.2f} HEDGE<extra></extra>',
        text=x_labels
    ),
    row=1, col=1
)

# Add zero line for PnL
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# 2. Hedge Ratio
colors = ['red' if x < 0 else 'green' for x in display_df['hedge_ratio']]
fig.add_trace(
    go.Bar(
        x=x_vals,
        y=display_df['hedge_ratio'],
        name='Hedge Ratio',
        marker=dict(color=colors),
        hovertemplate='Time: %{text}<br>Ratio: %{y}<extra></extra>',
        text=x_labels
    ),
    row=1, col=2
)

# 3. SM Delta with 95% CI
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=display_df['sm_delta'],
        mode='lines+markers',
        name='SM Delta',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=6),
        hovertemplate='Time: %{text}<br>Delta: %{y:.4f}<extra></extra>',
        text=x_labels
    ),
    row=2, col=1
)

# Add ±0.5 reference lines for delta
fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
fig.add_hline(y=-0.5, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# 4. Smile Slope with Uncertainty
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=display_df['slope_mean'],
        mode='lines',
        name='Slope Mean',
        line=dict(color='#AB63FA', width=3),
        hovertemplate='Time: %{text}<br>Slope: %{y:.6f}<extra></extra>',
        text=x_labels
    ),
    row=2, col=2
)

# Add 95% confidence interval
upper = display_df['slope_mean'] + 1.96 * display_df['slope_std']
lower = display_df['slope_mean'] - 1.96 * display_df['slope_std']

fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(171,99,250,0.2)',
        name='95% CI',
        hoverinfo='skip'
    ),
    row=2, col=2
)

# 5. BTC Price
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=display_df['btc_price'],
        mode='lines+markers',
        name='ETH Price',
        line=dict(color='#FFA15A', width=3),
        marker=dict(size=6),
        hovertemplate='Time: %{text}<br>Price: $%{y:,.2f}<extra></extra>',
        text=x_labels
    ),
    row=3, col=1
)

# 6. Volatility
fig.add_trace(
    go.Scatter(
        x=x_vals,
        y=display_df['volatility'],
        mode='lines+markers',
        name='Volatility',
        line=dict(color='#19D3F3', width=3),
        marker=dict(size=6),
        hovertemplate='Time: %{text}<br>Vol: %{y:.4f}<extra></extra>',
        text=x_labels
    ),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=1000,
    showlegend=True,
    template="plotly_white",
    hovermode='x unified',
    font=dict(size=10)
)

# Update axes labels
fig.update_xaxes(title_text="Execution Index", row=3, col=1)
fig.update_xaxes(title_text="Execution Index", row=3, col=2)

fig.update_yaxes(title_text="PnL (HEDGE)", row=1, col=1)
fig.update_yaxes(title_text="Hedge Ratio", row=1, col=2)
fig.update_yaxes(title_text="Delta", row=2, col=1)
fig.update_yaxes(title_text="Slope", row=2, col=2)
fig.update_yaxes(title_text="Price (USD)", row=3, col=1)
fig.update_yaxes(title_text="Volatility", row=3, col=2)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# DATA TABLE
# ==============================================================================
st.subheader("📝 Execution Log")

# Format data for display
display_table = display_df[[
    'timestamp_display', 'btc_price', 'sm_delta', 
    'slope_mean', 'slope_std', 'hedge_ratio', 'volatility', 'pnl'
]].copy()

display_table.columns = [
    'Time', 'ETH Price', 'SM Delta', 
    'Slope μ', 'Slope σ', 'Hedge Ratio', 'Volatility', 'PnL'
]

# Style the dataframe
st.dataframe(
    display_table.tail(20).style.format({
        'ETH Price': '${:,.2f}',
        'SM Delta': '{:.4f}',
        'Slope μ': '{:.6f}',
        'Slope σ': '{:.6f}',
        'Hedge Ratio': '{:.0f}',
        'Volatility': '{:.4f}',
        'PnL': '{:+.2f}'
    }).background_gradient(subset=['PnL'], cmap='RdYlGn'),
    use_container_width=True
)

# Download button
csv_data = log_df.to_csv(index=False)
st.download_button(
    label="📥 Download Full Log",
    data=csv_data,
    file_name=f"volhedge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ==============================================================================
# VOLATILITY HISTORY (IF AVAILABLE)
# ==============================================================================
if not hist_df.empty:
    st.markdown("---")
    st.subheader("📈 Historical Volatility Smile (Last 30 Days)")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=hist_df['Dates'], 
        y=hist_df['ATM_IV'],
        mode='lines',
        name='ATM IV',
        line=dict(color='gray', width=2)
    ))
    
    fig2.add_trace(go.Scatter(
        x=hist_df['Dates'],
        y=hist_df['IV_25D_PUT'],
        mode='lines',
        name='25Δ Put IV',
        line=dict(color='red', width=2)
    ))
    
    fig2.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Implied Volatility",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
st.markdown("---")
st.subheader("📊 Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**PnL Statistics**")
    st.write(f"Max: {display_df['pnl'].max():.2f}")
    st.write(f"Min: {display_df['pnl'].min():.2f}")
    st.write(f"Mean: {display_df['pnl'].mean():.2f}")
    st.write(f"Std Dev: {display_df['pnl'].std():.2f}")

with col2:
    st.markdown("**Delta Statistics**")
    st.write(f"Max: {display_df['sm_delta'].max():.4f}")
    st.write(f"Min: {display_df['sm_delta'].min():.4f}")
    st.write(f"Mean: {display_df['sm_delta'].mean():.4f}")
    st.write(f"Std Dev: {display_df['sm_delta'].std():.4f}")

with col3:
    st.markdown("**Hedge Ratio Distribution**")
    st.write(f"Max: {int(display_df['hedge_ratio'].max())}")
    st.write(f"Min: {int(display_df['hedge_ratio'].min())}")
    st.write(f"Mode: {int(display_df['hedge_ratio'].mode()[0])}")
    st.write(f"Rebalances: {(display_df['hedge_ratio'].diff() != 0).sum()}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")

with col2:
    st.caption(f"📊 Total records: {len(log_df)}")

with col3:
    st.caption(f"📈 Displaying: Last {len(display_df)} points")

with col4:
    st.caption(f"🔄 Next refresh: {refresh_interval}s")

# Auto-refresh logic (non-blocking)
import time
placeholder = st.empty()
for i in range(refresh_interval, 0, -1):
    placeholder.caption(f"⏱️ Auto-refresh in {i}s...")
    time.sleep(1)
placeholder.empty()
st.rerun()