import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
import psycopg2
from dotenv import load_dotenv
import os

# Must be the first Streamlit command
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()

# Database connection parameters
db_params = {
    'host': os.getenv('host'),
    'port': os.getenv('port'),
    'database': os.getenv('dbname'),
    'user': os.getenv('user'),
    'password': os.getenv('password')
}

# Custom CSS for the table
st.markdown("""
<style>
    .stDataFrame {
        width: 100%;
    }
    .stDataFrame table {
        width: 100% !important;
    }
    .stDataFrame th {
        background-color: #0e1117;
        color: white;
        font-weight: bold;
        text-align: left;
        padding: 12px;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .stDataFrame td {
        padding: 8px;
        border-bottom: 1px solid #30363d;
    }
    .stDataFrame tr:hover {
        background-color: #1e1e1e;
    }
    /* Custom styling for filters */
    .filter-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .filter-title {
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
    }
    /* Custom styling for metrics */
    .metric-value {
        text-align: right;
        font-family: monospace;
    }
    .metric-header {
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(**db_params)

def create_funnel_chart(funnel_data):
    """Create a vertical funnel chart using Plotly"""
    stages = [
        "Applications Received",
        "First Client Signup",
        "First Client Deposit",
        "First Trade",
        "First Earnings"
    ]
    
    values = [
        funnel_data['total_applications'],
        funnel_data['signup_activations'],
        funnel_data['deposit_activations'],
        funnel_data['trade_activations'],
        funnel_data['earning_activations']
    ]
    
    # Calculate conversion rates
    conversion_rates = []
    for i in range(1, len(values)):
        rate = (values[i] / values[i-1] * 100) if values[i-1] > 0 else 0
        conversion_rates.append(f"{rate:.1f}%")
    
    # Create the funnel chart
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        textposition="auto",
        textfont=dict(size=14),
        marker=dict(
            color=[
                "#1f77b4",  # Blue
                "#2ca02c",  # Green
                "#ff7f0e",  # Orange
                "#d62728",  # Red
                "#9467bd"   # Purple
            ]
        )
    ))
    
    # Update layout with title including country info if filtered
    title_text = "Partner Activation Funnel"
    if 'partner_country' in funnel_data and funnel_data['partner_country'] is not None:
        title_text += f" - {funnel_data['partner_country']}"
    
    fig.update_layout(
        title={
            'text': title_text,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(size=12),
        height=600,
        showlegend=False
    )
    
    return fig, conversion_rates

def fetch_funnel_metrics(conn, start_date, end_date, selected_country=None):
    """Fetch funnel metrics data from the database"""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # First get available countries for the filter
            countries_query = """
            SELECT DISTINCT partner_country
            FROM partner.partner_info
            WHERE is_internal = FALSE
                AND partner_country IS NOT NULL
            ORDER BY partner_country;
            """
            cursor.execute(countries_query)
            available_countries = [row['partner_country'] for row in cursor.fetchall()]
            
            # Calculate previous period for comparison
            period_length = (end_date - start_date).days
            prev_start_date = start_date - timedelta(days=period_length)
            prev_end_date = start_date - timedelta(days=1)
            
            # Initialize parameters dictionary
            params = {
                'start_date1': start_date,
                'end_date1': end_date,
                'start_date2': start_date,
                'end_date2': end_date,
                'prev_start_date': prev_start_date,
                'prev_end_date': prev_end_date
            }
            
            # Add country filter if country is selected
            country_filter = ""
            if selected_country:
                country_filter = "AND partner_country = %(selected_country)s"
                params['selected_country'] = selected_country
            
            # Base query with country filter
            overall_query = """
            WITH partner_clients AS (
                SELECT 
                    p.partner_id,
                    p.partner_country,
                    p.date_joined as partner_joined_date,
                    c.binary_user_id,
                    c.real_joined_date as first_client_joined_date,
                    c.first_deposit_date,
                    c.first_trade_date,
                    CASE WHEN c.real_joined_date IS NOT NULL THEN 
                        DATE_PART('day', c.real_joined_date::timestamp - p.date_joined::timestamp)
                    END as days_to_activation,
                    CASE WHEN c.real_joined_date >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END as is_active_last_30d,
                    COALESCE(e.has_earnings, 0) as has_earnings
                FROM partner.partner_info p
                LEFT JOIN client.staging_user_profile c 
                    ON c.affiliated_partner_id = p.partner_id 
                    AND c.is_internal = FALSE
                LEFT JOIN (
                    SELECT 
                        partner_id,
                        CASE WHEN SUM(total_earnings) > 0 THEN 1 ELSE 0 END as has_earnings
                    FROM partner.commission_daily
                    WHERE report_date BETWEEN %(start_date1)s AND %(end_date1)s
                    GROUP BY partner_id
                ) e ON e.partner_id = p.partner_id
                WHERE p.is_internal = FALSE
                    AND p.date_joined BETWEEN %(start_date2)s AND %(end_date2)s
                    {country_filter}
            ),
            previous_period_metrics AS (
                SELECT 
                    COUNT(DISTINCT partner_id) as prev_total_applications
                FROM partner.partner_info
                WHERE is_internal = FALSE
                    AND date_joined BETWEEN %(prev_start_date)s AND %(prev_end_date)s
                    {country_filter}
            ),
            current_period_metrics AS (
                SELECT 
                    COUNT(DISTINCT partner_id) as total_applications,
                    COUNT(DISTINCT CASE WHEN first_client_joined_date IS NOT NULL THEN partner_id END) as signup_activations,
                    COUNT(DISTINCT CASE WHEN first_deposit_date IS NOT NULL THEN partner_id END) as deposit_activations,
                    COUNT(DISTINCT CASE WHEN first_trade_date IS NOT NULL THEN partner_id END) as trade_activations,
                    COUNT(DISTINCT CASE WHEN has_earnings = 1 THEN partner_id END) as earning_activations,
                    AVG(days_to_activation) as avg_days_to_activation,
                    COUNT(DISTINCT CASE WHEN is_active_last_30d = 1 THEN partner_id END) as active_partners_30d
                FROM partner_clients
            )
            SELECT 
                cm.*,
                pm.prev_total_applications,
                ROUND(CAST(cm.signup_activations AS NUMERIC) / NULLIF(cm.total_applications, 0) * 100, 1) as activation_rate,
                ROUND(CAST(cm.active_partners_30d AS NUMERIC) / NULLIF(cm.total_applications, 0) * 100, 1) as active_partners_rate,
                ROUND(CAST((cm.total_applications - pm.prev_total_applications) AS NUMERIC) / NULLIF(pm.prev_total_applications, 0) * 100, 1) as application_growth_rate
            FROM current_period_metrics cm
            CROSS JOIN previous_period_metrics pm;
            """
            
            # Format query with country filter
            query = overall_query.format(country_filter=country_filter)
            
            # Execute overall metrics query
            cursor.execute(query, params)
            overall_df = pd.DataFrame(cursor.fetchall())
            
            # Query for country performance (unfiltered)
            country_query = """
            WITH partner_clients AS (
                SELECT 
                    p.partner_id,
                    p.partner_country,
                    p.date_joined as partner_joined_date,
                    c.binary_user_id,
                    c.real_joined_date as first_client_joined_date,
                    c.first_deposit_date,
                    CASE WHEN c.real_joined_date IS NOT NULL THEN 
                        DATE_PART('day', c.real_joined_date::timestamp - p.date_joined::timestamp)
                    END as days_to_activation,
                    CASE WHEN c.real_joined_date >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END as is_active_last_30d
                FROM partner.partner_info p
                LEFT JOIN client.staging_user_profile c 
                    ON c.affiliated_partner_id = p.partner_id 
                    AND c.is_internal = FALSE
                WHERE p.is_internal = FALSE
                    AND p.date_joined BETWEEN %(start_date1)s AND %(end_date1)s
                    AND p.partner_country IS NOT NULL
            ),
            country_metrics AS (
                SELECT 
                    partner_country,
                    COUNT(DISTINCT partner_id) as total_applications,
                    COUNT(DISTINCT CASE WHEN first_client_joined_date IS NOT NULL THEN partner_id END) as activated_partners,
                    AVG(days_to_activation) as avg_days_to_activation,
                    ROUND(
                        CAST(COUNT(DISTINCT CASE WHEN is_active_last_30d = 1 THEN partner_id END) AS NUMERIC) /
                        NULLIF(COUNT(DISTINCT CASE WHEN first_client_joined_date IS NOT NULL THEN partner_id END), 0) * 100,
                        1
                    ) as retention_rate,
                    ROUND(
                        CAST(COUNT(DISTINCT CASE WHEN first_client_joined_date IS NOT NULL THEN partner_id END) AS NUMERIC) /
                        NULLIF(COUNT(DISTINCT partner_id), 0) * 100,
                        1
                    ) as activation_rate
                FROM partner_clients
                GROUP BY partner_country
                HAVING COUNT(DISTINCT partner_id) >= 5
            )
            SELECT *,
                RANK() OVER (ORDER BY activation_rate DESC) as rank_by_activation,
                RANK() OVER (ORDER BY retention_rate DESC) as rank_by_retention
            FROM country_metrics
            ORDER BY total_applications DESC;
            """
            
            cursor.execute(country_query, params)
            country_df = pd.DataFrame(cursor.fetchall())
            
            return overall_df, country_df, available_countries
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), []

def format_metric(value, format_str=".1f", suffix=""):
    """Format metric values for display"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{value:{format_str}}{suffix}"
    except (ValueError, TypeError):
        return "N/A"

def format_country_table(df):
    """Format country performance table with styling"""
    formatted_df = df.copy()
    
    # Format numeric columns with NULL handling
    formatted_df['total_applications'] = formatted_df['total_applications'].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "-")
    formatted_df['activation_rate'] = formatted_df['activation_rate'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
    formatted_df['avg_days_to_activation'] = formatted_df['avg_days_to_activation'].apply(lambda x: f"{x:.1f} days" if pd.notnull(x) else "-")
    formatted_df['retention_rate'] = formatted_df['retention_rate'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
    
    # Select and rename columns
    return formatted_df[[
        'partner_country',
        'total_applications',
        'activation_rate',
        'avg_days_to_activation',
        'retention_rate'
    ]].rename(columns={
        'partner_country': 'Country',
        'total_applications': 'Applications',
        'activation_rate': 'Activation Rate',
        'avg_days_to_activation': 'Time to First Client',
        'retention_rate': 'Retention Rate'
    })

def display_dashboards(conn):
    """Display the dashboards tab content"""
    st.subheader("Partner Performance Dashboard")
    
    # Date range and country filter selection
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            key="dashboard_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="dashboard_end_date"
        )
    
    # Fetch initial data to get available countries
    overall_df, country_df, available_countries = fetch_funnel_metrics(conn, start_date, end_date)
    
    with col3:
        selected_country = st.selectbox(
            "Filter by Country",
            options=["All Countries"] + available_countries,
            help="Select a country to filter the data"
        )
    
    # Fetch filtered data if country is selected
    if selected_country != "All Countries":
        overall_df, country_df, _ = fetch_funnel_metrics(conn, start_date, end_date, selected_country)
    
    if overall_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Display KPI Cards in 2x2 grid
    col1, col2 = st.columns(2)
    
    # Row 1
    with col1:
        total_apps = overall_df['total_applications'].iloc[0]
        growth_rate = overall_df['application_growth_rate'].iloc[0]
        st.metric(
            "Total Applications",
            format_metric(total_apps, ",d", ""),
            format_metric(growth_rate, "+.1f", "% vs previous period") if not pd.isna(growth_rate) else "N/A",
            help="Total number of partner applications in the selected period"
        )
    
    with col2:
        activation_rate = overall_df['activation_rate'].iloc[0]
        st.metric(
            "Overall Activation Rate",
            format_metric(activation_rate, ".1f", "%"),
            help="Percentage of partners who have at least one active client"
        )
    
    # Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        avg_days = overall_df['avg_days_to_activation'].iloc[0]
        st.metric(
            "Average Time to First Activation",
            format_metric(avg_days, ".1f", " days"),
            help="Average number of days between partner signup and first client activation"
        )
    
    with col4:
        active_rate = overall_df['active_partners_rate'].iloc[0]
        st.metric(
            "Active Partners",
            format_metric(active_rate, ".1f", "%"),
            help="Percentage of partners who are currently active (had client activity in last 30 days)"
        )
    
    # Add spacing
    st.write("")
    st.write("")
    
    # Create and display funnel chart
    if selected_country != "All Countries":
        # If single country is selected, show that country's funnel
        country_funnel_data = overall_df.iloc[0]
        country_funnel_data['partner_country'] = selected_country
        funnel_fig, conversion_rates = create_funnel_chart(country_funnel_data)
    else:
        # Show overall funnel for multiple or no countries selected
        funnel_fig, conversion_rates = create_funnel_chart(overall_df.iloc[0])
    
    st.plotly_chart(funnel_fig, use_container_width=True)
    
    # Display conversion rates between stages
    st.subheader("Stage-by-Stage Conversion Rates")
    stages = [
        "Applications → First Client",
        "First Client → First Deposit",
        "First Deposit → First Trade",
        "First Trade → First Earnings"
    ]
    
    cols = st.columns(len(stages))
    for col, stage, rate in zip(cols, stages, conversion_rates):
        with col:
            st.metric(
                stage,
                rate if rate != "nan%" else "N/A",
                help="Conversion rate between consecutive stages"
            )
    
    # Add spacing before geographical section
    st.write("")
    st.write("")
    
    # Geographical Performance Section
    st.subheader("Geographical Performance Analysis")
    
    if country_df.empty:
        st.warning("No country-specific data available for the selected date range.")
        return
    
    # Top 10 and Bottom 10 performing countries (using unfiltered data)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Countries")
        top_countries = country_df.nsmallest(10, 'rank_by_activation')
        st.dataframe(
            format_country_table(top_countries),
            hide_index=True,
            use_container_width=True
        )
        
    with col2:
        st.subheader("Bottom Performing Countries")
        bottom_countries = country_df.nlargest(10, 'rank_by_activation')
        st.dataframe(
            format_country_table(bottom_countries),
            hide_index=True,
            use_container_width=True
        )
    
    # Add spacing
    st.write("")
    
    # Full country performance table with sorting options
    st.subheader("Complete Country Performance Table")
    if selected_country != "All Countries":
        st.info(f"Showing data for selected country: {selected_country}")
        # Filter country_df for selected country
        filtered_df = country_df[country_df['partner_country'] == selected_country]
    else:
        filtered_df = country_df
    
    # Sorting options
    sort_col = st.selectbox(
        "Sort by",
        options=[
            "Applications",
            "Activation Rate",
            "Time to First Client",
            "Retention Rate"
        ],
        key="country_sort"
    )
    
    # Map selected column to actual column name
    sort_map = {
        "Applications": "total_applications",
        "Activation Rate": "activation_rate",
        "Time to First Client": "avg_days_to_activation",
        "Retention Rate": "retention_rate"
    }
    
    # Sort the filtered dataframe
    sorted_df = filtered_df.sort_values(
        sort_map[sort_col],
        ascending=False if sort_col == "Applications" else True
    )
    
    # Display filtered table
    st.dataframe(
        format_country_table(sorted_df),
        hide_index=True,
        use_container_width=True
    )
    
    # Add export button for filtered data
    if not sorted_df.empty:
        csv = format_country_table(sorted_df).to_csv(index=False)
        st.download_button(
            label="Export Country Performance Data",
            data=csv,
            file_name="country_performance.csv",
            mime="text/csv"
        )

def main():
    st.title("Partner Performance Dashboard")
    
    try:
        # Create database connection
        conn = get_db_connection()
        
        # Display the dashboard
        display_dashboards(conn)
        
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 