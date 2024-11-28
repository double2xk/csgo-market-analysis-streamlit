import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from datetime import datetime
# import numpy as np
# from scipy import stats

# Page configuration with improved styling
st.set_page_config(
    page_title="CS:GO Skins Market Analysis Dashboard 2023-2024",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading for improved performance
@st.cache_data
# Replace the load_data function with file upload handling
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Derived columns
            df['month_year'] = df['date'].dt.strftime('%Y-%m')
            df['day_of_week'] = df['date'].dt.day_name()
            df['hour'] = pd.to_datetime(df['unix timestamp'].astype(float), unit='s').dt.hour    
           
            # Calculate price per quantity
            df['price_per_item'] = (df['total_value'] / df['quantity']).round(2)
            
            # Add wear condition from item name
            wear_conditions = ['Factory New', 'Minimal Wear', 'Field-Tested', 'Well-Worn', 'Battle-Scarred']
            df['wear_condition'] = df['item_name'].str.extract(f'({"|".join(wear_conditions)})')
            
            return df
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    return None


# File upload section at the top of the app
st.title("🎮 CS:GO Market Analysis Dashboard")
uploaded_file = st.file_uploader("Upload provided 'csgo-sales-data-2023-2024.csv' file", type=['csv'])


# Only show the dashboard if a file is uploaded and processed successfully
if uploaded_file is not None:
    df = process_uploaded_file(uploaded_file)
    
    if df is not None:
        st.markdown("""
            This dashboard provides comprehensive analysis of CS:GO market sales data.
            The data has been filtered to use items which cost is more than $1 to exclude overwhelming amount of cheap items.
            Use the filters in the sidebar to explore different aspects of the market.
        """)

        # Sidebar with filters
        st.sidebar.title("🎮 Analysis Filters")

        # Date range filter with default of full data range minus one month
        default_end_date = df['date'].max().replace(day=1) - pd.DateOffset(months=1)
        default_start_date = df['date'].min()
        date_range = st.sidebar.date_input(
            "📅 Select Date Range",
            value=(default_start_date, default_end_date),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )

        # Advanced filters in an expander
        with st.sidebar.expander("🔍 Advanced Filters"):
            # Price range filter with logarithmic scale
            price_range = st.slider(
                "💰 Price Range ($)",
                min_value=float(df['price'].min()),
                max_value=float(df['price'].max()),
                value=(float(df['price'].min()), float(df['price'].max())),
                format="$%.2f"
            )
            
            # Category filter with counts
            category_counts = df['category_name'].value_counts()
            category_options = [f"{cat} ({count:,})" for cat, count in category_counts.items()]
            selected_categories = st.multiselect(
                "🎯 Weapon Categories",
                options=category_options,
                default=category_options
            )
            selected_categories = [cat.split(" (")[0] for cat in selected_categories]
            
            # Rarity filter with color coding
            rarity_options = df['rarity_name'].unique()
            selected_rarities = st.multiselect(
                "✨ Item Rarities",
                options=rarity_options,
                default=rarity_options
            )
            
            # Wear condition filter
            wear_conditions = df['wear_condition'].dropna().unique()
            selected_wear = st.multiselect(
                "🔧 Wear Conditions",
                options=wear_conditions,
                default=wear_conditions
            )

        # Filter data based on all selections
        filtered_df = df[
            (df['date'].dt.date >= date_range[0]) &
            (df['date'].dt.date <= date_range[1]) &
            (df['category_name'].isin(selected_categories)) &
            (df['rarity_name'].isin(selected_rarities)) &
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1]) &
            (df['wear_condition'].isin(selected_wear))
        ]

        # Dashboard Header
        st.title("🎮 CS:GO Market Analysis Dashboard")
        st.markdown("""
            This dashboard provides comprehensive analysis of CS:GO market sales data from June 2023 to June 2024.
            The data has been filtered to use items which cost is more than $1 to exclude overwhelming amount of cheap items.
            Use the filters in the sidebar to explore different aspects of the market.
        """)

        # Enhanced metrics with trend indicators
        def calculate_trend(current, previous):
            if previous == 0:
                return 0
            return ((current - previous) / previous) * 100

        # Get the first and last month in the selected date range
        latest_month_start = filtered_df['date'].max().replace(day=1)
        previous_month_start = latest_month_start - pd.DateOffset(months=1)

        # Get data for comparison
        current_month_df = df[df['date'].dt.to_period('M') == latest_month_start.to_period('M')]
        previous_month_df = df[df['date'].dt.to_period('M') == previous_month_start.to_period('M')]

        # Calculate metrics with trends
        current_sales = current_month_df['total_value'].sum()
        previous_sales = previous_month_df['total_value'].sum()
        sales_trend = calculate_trend(current_sales, previous_sales)

        current_avg_price = current_month_df['price'].mean()
        previous_avg_price = previous_month_df['price'].mean()
        price_trend = calculate_trend(current_avg_price, previous_avg_price)

        # Add volume trend
        current_volume = current_month_df['quantity'].sum()
        previous_volume = previous_month_df['quantity'].sum()
        volume_trend = calculate_trend(current_volume, previous_volume)

        total_average_price = filtered_df['price'].mean()
        total_items_sold = filtered_df['quantity'].sum()

        # Display enhanced metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Sales Volume",
                f"${filtered_df['total_value'].sum():,.2f}",
                f"{sales_trend:+.1f}% Month on month",
                help=f"(Month on month statistic) Comparing {latest_month_start.strftime('%B %Y')} to {previous_month_start.strftime('%B %Y')}"
            )
        with col2:
            st.metric(
                "Average Item Price",
                f"${filtered_df['price'].mean():,.2f}",
                f"{price_trend:+.1f}% Month on month",
                help=f"(Month on month statistic) Comparing {latest_month_start.strftime('%B %Y')} to {previous_month_start.strftime('%B %Y')}"
            )
        with col3:
            st.metric(
                "Total Items Sold",
                f"{filtered_df['quantity'].sum():,}",
                f"{volume_trend:+.1f}% Month on month",
                help=f"(Month on month statistic) Comparing {latest_month_start.strftime('%B %Y')} to {previous_month_start.strftime('%B %Y')}"
            )
        with col4:
            st.metric(
                "Unique Items",
                f"{filtered_df['item_name'].nunique():,}",
                help="Number of distinct items traded in selected period"
            )

        # Market Overview Section
        st.header("📊 Market Overview")
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎯 Categories", "💰 Price Analysis"])

        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced daily sales trend with moving average
                daily_sales = filtered_df.groupby('date')['total_value'].sum().reset_index()
                daily_sales['MA7'] = daily_sales['total_value'].rolling(7).mean()
                
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Scatter(
                    x=daily_sales['date'],
                    y=daily_sales['total_value'],
                    name="Daily Sales",
                    mode="lines",
                    line=dict(color="#1f77b4", width=1)
                ))
                fig_daily.add_trace(go.Scatter(
                    x=daily_sales['date'],
                    y=daily_sales['MA7'],
                    name="7-day Moving Average",
                    line=dict(color="#ff7f0e", width=2)
                ))
                fig_daily.update_layout(
                    title="Daily Sales Volume with Trend",
                    xaxis_title="Date",
                    yaxis_title="Sales Volume ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                # Hourly sales pattern
                hourly_sales = filtered_df.groupby('hour')['total_value'].mean().reset_index()
                fig_hourly = px.line(
                    hourly_sales,
                    x='hour',
                    y='total_value',
                    title="Average Sales by Hour of Day",
                    labels={'total_value': 'Average Sales ($)', 'hour': 'Hour'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced category distribution
                category_sales = filtered_df.groupby('category_name').agg({
                    'total_value': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                category_sales['avg_price'] = (category_sales['total_value'] / category_sales['quantity']).round(2)
                
                fig_category = px.treemap(
                    category_sales,
                    path=['category_name'],
                    values='total_value',
                    color='avg_price',
                    title="Sales Distribution by Category",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_category, use_container_width=True)
            
            with col2:
                # Rarity analysis
                rarity_sales = filtered_df.groupby(['rarity_name', 'rarity_color']).agg({
                    'total_value': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                rarity_sales['avg_price'] = (rarity_sales['total_value'] / rarity_sales['quantity']).round(2)
                
                fig_rarity = px.sunburst(
                    rarity_sales,
                    path=['rarity_name'],
                    values='total_value',
                    color='avg_price',
                    title="Sales Distribution by Rarity",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_rarity, use_container_width=True)

        with tab3:
            # Price distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution histogram
                fig_price_dist = px.histogram(
                    filtered_df,
                    x='price',
                    nbins=50,
                    title="Price Distribution",
                    labels={'price': 'Price ($)', 'count': 'Number of Sales'}
                )
                fig_price_dist.update_layout(showlegend=False)
                st.plotly_chart(fig_price_dist, use_container_width=True)
            
            with col2:
                # Price trends by wear condition
                wear_price = filtered_df.groupby(['wear_condition', 'date'])['price'].mean().reset_index()
                fig_wear = px.line(
                    wear_price,
                    x='date',
                    y='price',
                    color='wear_condition',
                    title="Average Price by Wear Condition",
                    labels={'price': 'Average Price ($)', 'date': 'Date'}
                )
                st.plotly_chart(fig_wear, use_container_width=True)

        # Market Items Analysis
        st.header("🎯 Market Items Analysis")

        # Top items analysis with enhanced metrics
        top_items = filtered_df.groupby('item_name').agg({
            'total_value': 'sum',
            'quantity': 'sum',
            'price': ['mean', 'std'],
            'category_name': 'first',
            'rarity_name': 'first'
        }).reset_index()

        # Flatten the multi-index columns
        top_items.columns = ['item_name', 'total_value', 'quantity', 'price_mean', 'price_std', 'category_name', 'rarity_name']

        # Calculate coefficient of variation
        top_items['price_cv'] = top_items['price_std'] / top_items['price_mean']
        top_items = top_items.sort_values('total_value', ascending=False).head(10)

        # Display top items table with enhanced formatting
        st.subheader("📊 Top Selling Items")
        formatted_top_items = pd.DataFrame({
            'Item Name': top_items['item_name'],
            'Total Sales': top_items['total_value'].map('${:,.2f}'.format),
            'Quantity': top_items['quantity'].map('{:,}'.format),
            'Avg Price': top_items['price_mean'].map('${:,.2f}'.format),
            'Price Volatility': top_items['price_cv'].map('{:.2%}'.format)
        })
        st.dataframe(formatted_top_items, hide_index=True)

        # Market Volatility Analysis
        st.header("📈 Market Volatility Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Daily price volatility
            daily_volatility = filtered_df.groupby('date')['price'].agg(['mean', 'std']).reset_index()
            daily_volatility['cv'] = daily_volatility['std'] / daily_volatility['mean']
            
            fig_vol = px.line(
                daily_volatility,
                x='date',
                y='cv',
                title="Daily Price Volatility",
                labels={'cv': 'Coefficient of Variation', 'date': 'Date'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            # Category volatility comparison
            category_volatility = filtered_df.groupby('category_name')['price'].agg(['mean', 'std']).reset_index()
            category_volatility['cv'] = category_volatility['std'] / category_volatility['mean']
            
            fig_cat_vol = px.bar(
                category_volatility.sort_values('cv'),
                x='category_name',
                y='cv',
                title="Price Volatility by Category",
                labels={'cv': 'Coefficient of Variation', 'category_name': 'Category'}
            )
            st.plotly_chart(fig_cat_vol, use_container_width=True)

        # Market Insights Section
        st.header("💡 Market Insights")

        # Calculate and display key insights
        total_market_value = filtered_df['total_value'].sum()
        avg_daily_volume = filtered_df.groupby('date')['total_value'].sum().mean()
        most_traded_category = filtered_df.groupby('category_name')['quantity'].sum().idxmax() # Most traded category
        highest_value_rarity = filtered_df.groupby('rarity_name')['total_value'].sum().idxmax() # Most valuable rarity

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
                📊 Market Overview:
                - Total Market Value: ${total_market_value:,.2f}
                - Average Daily Trading Volume: ${avg_daily_volume:,.2f}
                - Most Traded Category: {most_traded_category}
                - Highest Value Rarity: {highest_value_rarity}
            """)

        with col2:
            # Calculate market concentration (HHI)
            market_shares = filtered_df.groupby('item_name')['total_value'].sum() / total_market_value
            hhi = (market_shares ** 2).sum()
            
            st.info(f"""
                🎯 Market Concentration:
                - Market Concentration (HHI): {hhi:.4f}
                - Market Type: {'Highly Concentrated' if hhi > 0.25 else 'Moderately Concentrated' if hhi > 0.15 else 'Competitive'}
                - Top 10 Items Market Share: {(filtered_df.groupby('item_name')['total_value'].sum().nlargest(10).sum() / total_market_value):.1%}
            """)

        # Advanced Market Analysis
        st.header("🔍 Advanced Market Analysis")

        # Trading Pattern Analysis
        col1, col2 = st.columns(2)

        with col1:
            # Day of week analysis
            daily_pattern = filtered_df.groupby('day_of_week').agg({
                'total_value': 'sum',
                'quantity': 'sum'
            }).reset_index()
            # Ensure correct day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_pattern['day_of_week'] = pd.Categorical(daily_pattern['day_of_week'], categories=day_order, ordered=True)
            daily_pattern = daily_pattern.sort_values('day_of_week')
            
            fig_daily_pattern = px.bar(
                daily_pattern,
                x='day_of_week',
                y=['total_value', 'quantity'],
                title="Trading Patterns by Day of Week",
                barmode='group',
                labels={
                    'day_of_week': 'Day of Week',
                    'total_value': 'Total Value ($)',
                    'quantity': 'Quantity Sold'
                }
            )
            st.plotly_chart(fig_daily_pattern, use_container_width=True)

        with col2:
            # Monthly trend analysis
            monthly_trend = filtered_df.groupby('month_year').agg({
                'total_value': 'sum',
                'quantity': 'sum',
                'price': 'mean'
            }).reset_index()
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_trend['month_year'],
                y=monthly_trend['total_value'],
                name='Total Value',
                yaxis='y'
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_trend['month_year'],
                y=monthly_trend['price'],
                name='Average Price',
                yaxis='y2'
            ))
            fig_monthly.update_layout(
                title='Monthly Market Overview',
                yaxis=dict(title='Total Value ($)', side='left'),
                yaxis2=dict(title='Average Price ($)', side='right', overlaying='y'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        # StatTrak™ Analysis
        st.subheader("⚔️ StatTrak™ Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # StatTrak premium analysis
            # Ensure only to compare matching items (same item with and without StatTrak)
            stattrak_base_names = filtered_df[filtered_df['stattrak'] == 1]['item_name'].str.replace('StatTrak™ ', '', regex=False)
            matching_items = filtered_df[
                (filtered_df['stattrak'] == 0) & 
                (filtered_df['item_name'].isin(stattrak_base_names))
            ]
            
            # Calculate average prices for matching items
            stattrak_prices = filtered_df[filtered_df['stattrak'] == 1].groupby('category_name')['price'].mean()
            normal_prices = matching_items.groupby('category_name')['price'].mean()
            
            # Combine into a DataFrame
            stattrak_analysis = pd.DataFrame({
                'category_name': stattrak_prices.index,
                'normal_price': normal_prices,
                'stattrak_price': stattrak_prices
            }).reset_index(drop=True)
            
            # Calculate premium percentage
            stattrak_analysis['premium_percentage'] = (
                (stattrak_analysis['stattrak_price'] - stattrak_analysis['normal_price']) 
                / stattrak_analysis['normal_price'] * 100
            )
            
            fig_stattrak = px.bar(
                stattrak_analysis,
                x='category_name',
                y='premium_percentage',
                title='StatTrak™ Premium by Category',
                labels={
                    'category_name': 'Category',
                    'premium_percentage': 'StatTrak™ Premium (%)'
                }
            )
            st.plotly_chart(fig_stattrak, use_container_width=True)

        with col2:
            # StatTrak™ market share
            stattrak_share = filtered_df.groupby('stattrak')['total_value'].sum()
            fig_stattrak_share = px.pie(
                values=stattrak_share.values,
                names=['Regular', 'StatTrak™'],
                title='Market Share: StatTrak™ vs Regular Items'
            )
            st.plotly_chart(fig_stattrak_share, use_container_width=True)

        # Price Range Distribution
        st.subheader("💰 Price Range Analysis")
        price_ranges = pd.cut( # Cut the data into price ranges
            filtered_df['price'],
            bins=[1, 5, 10, 50, 100, 500, float('inf')],
            labels=['$1-5', '$5-10', '$10-50', '$50-100', '$100-500', '$500+']
        )
        price_range_stats = filtered_df.groupby(price_ranges).agg({
            'total_value': 'sum',
            'quantity': 'sum',
            'item_name': 'nunique'
        }).reset_index()
        price_range_stats.columns = ['Price Range', 'Total Value', 'Quantity', 'Unique Items']

        col1, col2 = st.columns(2)

        with col1:
            fig_price_dist = px.bar(
                price_range_stats,
                x='Price Range',
                y=['Total Value', 'Quantity'],
                title='Distribution by Price Range',
                barmode='group'
            )
            st.plotly_chart(fig_price_dist, use_container_width=True)

        with col2:
            fig_unique_items = px.bar(
                price_range_stats,
                x='Price Range',
                y='Unique Items',
                title='Number of Unique Items by Price Range'
            )
            st.plotly_chart(fig_unique_items, use_container_width=True)

        # Wear Condition Analysis
        st.subheader("🔧 Wear Condition Impact")
        wear_analysis = filtered_df.groupby('wear_condition').agg({
            'price': 'mean',
            'quantity': 'sum',
            'total_value': 'sum'
        }).reset_index()

        # Rename columns for clarity
        wear_analysis.columns = ['wear_condition', 'avg_price', 'quantity', 'total_value']

        col1, col2 = st.columns(2)

        with col1:
            # Average price by wear
            fig_wear_price = px.bar(
                wear_analysis,
                x='wear_condition',
                y='avg_price',
                title='Average Price by Wear Condition',
                labels={
                    'wear_condition': 'Wear Condition',
                    'avg_price': 'Average Price ($)'
                }
            )
            fig_wear_price.update_layout(
                xaxis_title="Wear Condition",
                yaxis_title="Average Price ($)",
                xaxis={'categoryorder': 'array',
                    'categoryarray': ['Factory New', 'Minimal Wear', 
                                    'Field-Tested', 'Well-Worn', 'Battle-Scarred']}
            )
            st.plotly_chart(fig_wear_price, use_container_width=True)

        with col2:
            # Market share by wear
            fig_wear_share = px.pie(
                wear_analysis,
                values='total_value',
                names='wear_condition',
                title='Market Share by Wear Condition'
            )
            st.plotly_chart(fig_wear_share, use_container_width=True)

        # Data download option
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Download Data")

        # Convert filtered data to CSV
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_df)
        st.sidebar.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="csgo_market_data_filtered.csv",
            mime="text/csv"
        )

        # Footer with data freshness
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center'>
                <p>Data last updated: {}</p>
            </div>
        """.format(df['date'].max().strftime('%Y-%m-%d')), unsafe_allow_html=True)
            
    else:
        st.error("Please upload a valid CS:GO market data CSV file with the required columns.")
else:
    st.info("Please upload a CSV file to begin the analysis.")
    
    # Add example of expected CSV format
    st.markdown("""
    The file can be downloaded from the [Google Drive Shared Link](https://drive.google.com/file/d/1J4Jy7y010t6j-gR5dpmHNlajbukXz6Xe/view?usp=sharing)
    """)