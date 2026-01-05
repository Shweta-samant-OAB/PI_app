import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Currency conversion rate (USD to INR)
USD_TO_INR = 83.0  # You can update this rate as needed

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

# File upload section
st.header("📁 Upload Your Cloud Cost Data")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔷 Azure")
    azure_file = st.file_uploader("Upload Azure CSV", type=['csv'], key='azure')
    if azure_file:
        try:
            df = pd.read_csv(azure_file)
            # Azure uses ServiceName and UsageDate (date/month/year format)
            if 'ServiceName' in df.columns and 'UsageDate' in df.columns:
                # Find cost column
                cost_col = next((col for col in df.columns if 'cost' in col.lower()), None)
                if cost_col:
                    df['Cost'] = pd.to_numeric(df[cost_col], errors='coerce')
                    df = df[df['Cost'] > 0]
                    # Parse date (date/month/year format)
                    df['Date'] = pd.to_datetime(df['UsageDate'], format='%d/%m/%Y', errors='coerce')
                    df['Month'] = df['Date'].dt.to_period('M').astype(str)
                    st.session_state.azure_data = df
                    st.success(f"✅ {len(df)} Azure records loaded")
                else:
                    st.error("Could not find cost column in Azure CSV")
            else:
                st.error("Azure CSV must contain 'ServiceName' and 'UsageDate' columns")
        except Exception as e:
            st.error(f"Error reading Azure CSV: {str(e)}")

with col2:
    st.subheader("🔵 GCP")
    gcp_file = st.file_uploader("Upload GCP CSV", type=['csv'], key='gcp')
    if gcp_file:
        try:
            df = pd.read_csv(gcp_file)
            # GCP uses service_description and Subtotal
            if 'service_description' in df.columns and 'Subtotal' in df.columns:
                df['Cost_USD'] = pd.to_numeric(df['Subtotal'], errors='coerce')
                df['Cost_INR'] = df['Cost_USD'] * USD_TO_INR
                df = df[df['Cost_INR'] > 0]
                # Parse date if exists
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                if date_col:
                    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                    df['Month'] = df['Date'].dt.to_period('M').astype(str)
                st.session_state.gcp_data = df
                st.success(f"✅ {len(df)} GCP records loaded")
            else:
                st.error("GCP CSV must contain 'service_description' and 'Subtotal' columns")
        except Exception as e:
            st.error(f"Error reading GCP CSV: {str(e)}")

with col3:
    st.subheader("🟠 AWS")
    aws_file = st.file_uploader("Upload AWS CSV", type=['csv'], key='aws')
    if aws_file:
        try:
            df = pd.read_csv(aws_file)
            # AWS has Date column (month/date/year format) and Total costs($) column
            if 'Date' in df.columns and 'Total costs($)' in df.columns:
                # Parse date (month/date/year format)
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
                df['Month'] = df['Date'].dt.to_period('M').astype(str)
                
                # Convert Total costs from USD to INR
                df['Total_USD'] = pd.to_numeric(df['Total costs($)'], errors='coerce')
                df['Total_INR'] = df['Total_USD'] * USD_TO_INR
                
                # Convert all other service costs from USD to INR (excluding Date, Month, Total costs($))
                service_cols = [col for col in df.columns 
                              if col not in ['Date', 'Month', 'Total costs($)', 'Total_USD', 'Total_INR'] 
                              and pd.api.types.is_numeric_dtype(df[col])]
                
                for col in service_cols:
                    df[f'{col}_INR'] = pd.to_numeric(df[col], errors='coerce') * USD_TO_INR
                
                st.session_state.aws_data = df
                st.success(f"✅ {len(df)} AWS records loaded")
            else:
                st.error("AWS CSV must contain 'Date' and 'Total costs($)' columns")
        except Exception as e:
            st.error(f"Error reading AWS CSV: {str(e)}")

st.markdown("---")

# Check if any data is loaded
has_data = any([
    st.session_state.azure_data is not None,
    st.session_state.gcp_data is not None,
    st.session_state.aws_data is not None
])

if not has_data:
    st.info("👆 Please upload at least one CSV file to get started")
    st.markdown("""
    ### Expected CSV Formats:
    
    **Azure CSV should contain:**
    - **UsageDate** column (format: date/month/year, e.g., 15/03/2024)
    - **ServiceName** column
    - Cost column (costs in INR)
    
    **GCP CSV should contain:**
    - **service_description** column
    - **Subtotal** column (costs in USD, will be converted to INR)
    - Date column (optional)
    
    **AWS CSV should contain:**
    - **Date** column (format: month/date/year, e.g., 03/15/2024)
    - **Total costs($)** column (will be converted to INR)
    - Service columns (RDS, EC2, EKS, etc.) with costs in USD (will be converted to INR)
    """)
else:
    # Calculate totals (all in INR now)
    azure_total = st.session_state.azure_data['Cost'].sum() if st.session_state.azure_data is not None else 0
    gcp_total = st.session_state.gcp_data['Cost_INR'].sum() if st.session_state.gcp_data is not None else 0
    aws_total = st.session_state.aws_data['Total_INR'].sum() if st.session_state.aws_data is not None else 0
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
                value=f"₹{grand_total:,.2f}",
                delta="All Platforms"
            )
        
        with col2:
            if st.session_state.azure_data is not None:
                st.metric(
                    label="🔷 Azure",
                    value=f"₹{azure_total:,.2f}"
                )
            else:
                st.metric(label="🔷 Azure", value="No data")
        
        with col3:
            if st.session_state.gcp_data is not None:
                st.metric(
                    label="🔵 GCP",
                    value=f"₹{gcp_total:,.2f}"
                )
            else:
                st.metric(label="🔵 GCP", value="No data")
        
        with col4:
            if st.session_state.aws_data is not None:
                st.metric(
                    label="🟠 AWS",
                    value=f"₹{aws_total:,.2f}"
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
                    'Resources/Services': len(st.session_state.azure_data),
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
                top_services = service_costs.nlargest(10, 'Cost_INR')
                
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
