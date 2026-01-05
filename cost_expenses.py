import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Currency conversion rate (USD to INR)
USD_TO_INR = 90.25 

# Page configuration
st.set_page_config(
    page_title="Multi-Cloud Cost Dashboard",
    page_icon="☁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0078D4, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
""", unsafe_allow_html=True)

def format_inr(amount):
    if pd.isna(amount):
        return "₹0.00"
    s = f"{amount:,.2f}"
    parts = s.split(".")
    num = parts[0].replace(",", "")
    
    if len(num) > 3:
        last3 = num[-3:]
        rest = num[:-3]
        rest = ",".join([rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1])
        num = rest + "," + last3
    
    return f"₹{num}.{parts[1]}"

# Title
st.markdown('<h1 class="main-header">☁️ Multi-Cloud Cost Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Unified view of Azure, GCP, and AWS expenses (All costs in INR)**")
st.markdown("---")

# Currency converter info
with st.expander("ℹ️ Currency Conversion Info"):
    st.info(f"Exchange Rate: 1 USD = ₹{USD_TO_INR}")
    st.markdown("""
    - **AWS**: Costs converted from USD to INR
    - **GCP**: Costs converted from USD to INR
    - **Azure**: Costs already in INR
    """)

# Initialize session state for data storage
if 'azure_data' not in st.session_state:
    st.session_state.azure_data = None
if 'gcp_data' not in st.session_state:
    st.session_state.gcp_data = None
if 'aws_data' not in st.session_state:
    st.session_state.aws_data = None

# -------- AZURE --------
try:
    df = pd.read_csv("azure_costs.csv")

    required_cols = {'ServiceName', 'UsageDate'}
    if not required_cols.issubset(df.columns):
        st.error("Azure CSV must contain 'ServiceName' and 'UsageDate'")
    else:
        if 'Cost' in df.columns:
            df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
        elif 'CostUSD' in df.columns:
            df['Cost'] = pd.to_numeric(df['CostUSD'], errors='coerce') * USD_TO_INR
        else:
            st.error("Azure CSV must contain 'Cost' or 'CostUSD'")
            st.stop()

        df['Date'] = pd.to_datetime(df['UsageDate'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        st.session_state.azure_data = df

except FileNotFoundError:
    st.warning("azure_costs.csv not found")

# -------- GCP --------
try:
    df = pd.read_csv("gcp_costs.csv")

    if 'service_description' in df.columns and 'Subtotal' in df.columns:
        df['Cost_USD'] = pd.to_numeric(df['Subtotal'], errors='coerce')
        df['Cost_INR'] = df['Cost_USD'] * USD_TO_INR

        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['Month'] = df['Date'].dt.to_period('M').astype(str)

        st.session_state.gcp_data = df
    else:
        st.error("GCP CSV must contain 'service_description' and 'Subtotal'")

except FileNotFoundError:
    st.warning(" gcp_costs.csv not found")
# -------- AWS --------
try:
    df = pd.read_csv("aws_costs.csv")

    if 'Date' in df.columns and 'Total costs($)' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        df['Total_USD'] = pd.to_numeric(df['Total costs($)'], errors='coerce')
        df['Total_INR'] = df['Total_USD'] * USD_TO_INR

        service_cols = [
            col for col in df.columns
            if col not in ['Date', 'Month', 'Total costs($)', 'Total_USD', 'Total_INR']
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        for col in service_cols:
            df[f'{col}_INR'] = pd.to_numeric(df[col], errors='coerce') * USD_TO_INR

        st.session_state.aws_data = df
    else:
        st.error("AWS CSV must contain 'Date' and 'Total costs($)'")

except FileNotFoundError:
    st.warning("⚠️ aws_cost.csv not found")

st.markdown("---")

# Check if any data is loaded
has_data = any([
    st.session_state.azure_data is not None,
    st.session_state.gcp_data is not None,
    st.session_state.aws_data is not None
])

if not has_data:
    st.info("👆 Please upload at least one CSV file to get started")

else:
    # Calculate totals (all in INR)
    # Calculate totals (exact CSV columns + conversion)

    azure_total = pd.to_numeric(
        st.session_state.azure_data['Cost'], errors='coerce'
    ).sum()

    gcp_total = pd.to_numeric(
        st.session_state.gcp_data['Subtotal'], errors='coerce'
    ).sum() * USD_TO_INR
    print(gcp_total)

    aws_total = pd.to_numeric(
        st.session_state.aws_data['Total costs($)'], errors='coerce'
    ).sum() * USD_TO_INR
    print(aws_total)

    grand_total = azure_total + gcp_total + aws_total

    
    # Create tabs AFTER calculating totals
    tabs = st.tabs(["📊 Overview", "🔷 Azure", "🔵 GCP", "🟠 AWS"])

    # OVERVIEW TAB
    with tabs[0]:
        st.header("Cost Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Total Spend",
                value=format_inr(grand_total),
                delta="All Platforms"
            )
        
        with col2:
            if st.session_state.azure_data is not None:
                st.metric(
                    label="🔷 Azure",
                    value=format_inr(azure_total)
                )
            else:
                st.metric(label="🔷 Azure", value="No data")
        
        with col3:
            if st.session_state.gcp_data is not None:
                st.metric(
                    label="🔵 GCP",
                    value=format_inr(gcp_total)
                )
            else:
                st.metric(label="🔵 GCP", value="No data")
        
        with col4:
            if st.session_state.aws_data is not None:
                st.metric(
                    label="🟠 AWS",
                    value=format_inr(aws_total)
                )
            else:
                st.metric(label="🟠 AWS", value="No data")
        
        st.markdown("---")
        
        provider_data = []

        if azure_total > 0:
            provider_data.append({'Provider': 'Azure', 'Cost': azure_total})
        if gcp_total > 0:
            provider_data.append({'Provider': 'GCP', 'Cost': gcp_total})
        if aws_total > 0:
            provider_data.append({'Provider': 'AWS', 'Cost': aws_total})

        if provider_data:
            df_providers = pd.DataFrame(provider_data)

            fig = px.bar(
                df_providers,
                x='Provider',
                y='Cost',
                title='Cloud Provider Cost Comparison (INR)',
                color='Provider',
                color_discrete_map={
                    'Azure': '#00BFA5',
                    'GCP': '#4285F4',
                    'AWS': '#FF6600'
                },
                text='Cost'
            )

            fig.update_layout(
                showlegend=False,
                yaxis_title="Cost (₹)",
                xaxis_title="Cloud Provider"
            )

            fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')

            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📋 Quick Summary")
            summary_data = []
            
            if st.session_state.azure_data is not None:
                summary_data.append({
                    'Provider': '🔷 Azure',
                    'Total Cost': f"₹{azure_total:,.2f}",
                    'Records': len(st.session_state.azure_data),
                    '% of Total': f"{(azure_total/grand_total*100):.1f}%" if grand_total > 0 else "0%"
                })
            
            if st.session_state.gcp_data is not None:
                summary_data.append({
                    'Provider': '🔵 GCP',
                    'Total Cost': f"₹{gcp_total:,.2f}",
                    'Resources/Services': len(st.session_state.gcp_data),
                    '% of Total': f"{(gcp_total/grand_total*100):.1f}%" if grand_total > 0 else "0%"
                })
            
            if st.session_state.aws_data is not None:
                summary_data.append({
                    'Provider': '🟠 AWS',
                    'Total Cost': f"₹{aws_total:,.2f}",
                    'Resources/Services': len(st.session_state.aws_data),
                    '% of Total': f"{(aws_total/grand_total*100):.1f}%" if grand_total > 0 else "0%"
                })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # AZURE TAB
    with tabs[1]:
        if st.session_state.azure_data is not None:
            st.header(f"Azure Resources - Total: ₹{azure_total:,.2f}")

            if 'ServiceName' in st.session_state.azure_data.columns:
                st.subheader("Total Cost by Azure Service")

                service_costs = (
                    st.session_state.azure_data
                    .groupby('ServiceName', as_index=False)['Cost']
                    .sum()
                    .sort_values('Cost', ascending=False)
                )

                fig = px.bar(
                    service_costs,
                    x='Cost',
                    y='ServiceName',
                    orientation='h',
                    title='Azure Cost Incurred by Service',
                    labels={
                        'Cost': 'Total Cost (₹)',
                        'ServiceName': 'Azure Service'
                    },
                    color='ServiceName'
                )

                fig.update_layout(showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No Azure data uploaded yet")

    # GCP TAB
    with tabs[2]:
        if st.session_state.gcp_data is not None:
            st.header(f"GCP Services - Total: ₹{gcp_total:,.2f}")
            
            # Monthly cost trend by service (if date available)
            if 'Month' in st.session_state.gcp_data.columns and 'service_description' in st.session_state.gcp_data.columns:
                st.subheader("Monthly Cost Trend by Service")
                
                monthly_service = st.session_state.gcp_data.groupby(['Month', 'service_description'])['Cost_INR'].sum().reset_index()
                
                unique_services = monthly_service['service_description'].unique()
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
                color_map = {service: colors[i % len(colors)] for i, service in enumerate(unique_services)}
                
                fig = px.line(
                    monthly_service,
                    x='Month',
                    y='Cost_INR',
                    color='service_description',
                    title='GCP Monthly Cost by Service',
                    labels={'Cost_INR': 'Cost (₹)', 'service_description': 'Service'},
                    markers=True,
                    color_discrete_map=color_map
                )
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            # Top services by cost
            if 'service_description' in st.session_state.gcp_data.columns:
                st.subheader("Top 10 Services by Cost")
                service_costs = st.session_state.gcp_data.groupby('service_description')['Cost_INR'].sum().reset_index()
                top_services = (
                    service_costs
                    .nlargest(10, 'Cost_INR')
                    .sort_values('Cost_INR', ascending=True) 
                )
                
                
                fig = px.bar(
                    top_services,
                    x='Cost_INR',
                    y='service_description',
                    orientation='h',
                    title='Top 10 GCP Services by Cost',
                    labels={'Cost_INR': 'Cost (₹)', 'service_description': 'Service'},
                    color='Cost_INR',
                    color_continuous_scale='Teal'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No GCP data uploaded yet")

    # AWS TAB
    with tabs[3]:
        if st.session_state.aws_data is not None:
            st.header(f"AWS Services - Total: ₹{aws_total:,.2f}")
            
            st.subheader("Total Cost by Service")
            
            service_inr_cols = [col for col in st.session_state.aws_data.columns if col.endswith('_INR') and col != 'Total_INR']
            
            if service_inr_cols:
                service_totals = {}
                for col in service_inr_cols:
                    total = st.session_state.aws_data[col].sum()
                    if total > 0:
                        service_name = col.replace('_INR', '')
                        service_totals[service_name] = total
                
                if service_totals:
                    df_services = pd.DataFrame(list(service_totals.items()), columns=['Service', 'Cost_INR'])
                    df_services = df_services.sort_values('Cost_INR', ascending=False)
                    
                    colors_list = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
                    
                    fig = px.bar(
                        df_services,
                        x='Service',
                        y='Cost_INR',
                        title='AWS Cost by Service',
                        labels={'Cost_INR': 'Cost (₹)', 'Service': 'AWS Service'},
                        color='Service',
                        color_discrete_sequence=colors_list
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No AWS data uploaded yet")

# Footer
st.markdown("---")

