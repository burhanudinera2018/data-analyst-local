import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import sys
import hashlib
import json
import os

# ==================== PATH SETUP ====================
# Tambahkan parent directory ke sys.path untuk import scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import dari scripts
try:
    from scripts.ollama_helper import LocalLLM, get_llm_with_fallback, test_ollama_connection
except ImportError as e:
    st.error(f"Error importing ollama_helper: {e}")
    st.info("Pastikan struktur folder: data-analyst-local/scripts/ollama_helper.py")
    st.stop()

# ==================== SETUP ====================
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Inisialisasi cache di session state
if 'ai_cache' not in st.session_state:
    st.session_state.ai_cache = {}

# ==================== FUNGSI HELPER ====================

# Koneksi database
@st.cache_resource
def init_connection():
    return create_engine('postgresql://analyst:analyst123@localhost/ecommerce_analytics')

# Inisialisasi LLM
@st.cache_resource
def init_llm():
    """Inisialisasi LLM dengan fallback"""
    # Cek koneksi Ollama
    status = test_ollama_connection()
    if not status["connected"]:
        st.warning(f"⚠️ {status['error']}. Fitur AI Assistant mungkin tidak berfungsi.")
        return None
    
    # Tampilkan model yang tersedia
    if status["models"]:
        st.sidebar.success(f"✅ Ollama siap. Model: {', '.join(status['models'][:3])}")
    
    # Inisialisasi dengan model yang sesuai
    preferred_model = "gemma2:2b"  # Sesuaikan dengan model yang Anda miliki
    llm = get_llm_with_fallback(preferred_model)
    
    if llm is None:
        st.error("❌ Tidak ada model LLM yang tersedia. Jalankan 'ollama pull gemma2:2b' di terminal.")
        return None
    
    return llm

# Fungsi untuk menjalankan query dengan error handling
@st.cache_data(ttl=300)  # Cache 5 menit
def run_query(query):
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()

# Fungsi untuk mendapatkan RFM Segments
@st.cache_data(ttl=600)
def get_rfm_segments():
    query = """
    WITH rfm AS (
        SELECT 
            user_id,
            EXTRACT(DAY FROM AGE(NOW()::timestamp, MAX(created_at)::timestamp)) as recency,
            COUNT(DISTINCT order_id) as frequency,
            SUM(sale_price) as monetary
        FROM order_items
        WHERE status = 'Complete'
        GROUP BY user_id
    ),
    rfm_scores AS (
        SELECT 
            user_id,
            recency,
            frequency,
            monetary,
            NTILE(5) OVER (ORDER BY recency DESC) as r_score,
            NTILE(5) OVER (ORDER BY frequency) as f_score,
            NTILE(5) OVER (ORDER BY monetary) as m_score
        FROM rfm
    )
    SELECT 
        user_id,
        ROUND(recency::numeric, 2) as recency,
        ROUND(frequency::numeric, 2) as frequency,
        ROUND(monetary::numeric, 2) as monetary,
        CONCAT(r_score, f_score, m_score) as rfm_score,
        CASE 
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 4 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Potential Loyalists'
            WHEN r_score >= 4 AND f_score <= 2 AND m_score <= 2 THEN 'New Customers'
            WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Lost'
            ELSE 'Others'
        END as segment
    FROM rfm_scores
    ORDER BY monetary DESC
    """
    return run_query(query)

# Fungsi untuk mendapatkan data forecasting
@st.cache_data(ttl=300)
def get_forecast_data():
    query = """
    SELECT 
        DATE(created_at) as date,
        COUNT(DISTINCT order_id) as num_orders,
        SUM(sale_price) as total_sales,
        COUNT(DISTINCT user_id) as unique_customers
    FROM order_items
    WHERE status = 'Complete'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    df = run_query(query)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

# Fungsi untuk mendapatkan data penjualan per produk
@st.cache_data(ttl=300)
def get_product_sales(start_date, end_date):
    query = f"""
    SELECT 
        p.id,
        p.name as product_name,
        p.category,
        COUNT(DISTINCT oi.order_id) as num_orders,
        SUM(oi.quantity) as total_quantity,
        SUM(oi.sale_price) as total_revenue,
        AVG(oi.sale_price) as avg_price
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE DATE(oi.created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND oi.status = 'Complete'
    GROUP BY p.id, p.name, p.category
    ORDER BY total_revenue DESC
    """
    return run_query(query)

# Fungsi untuk mendapatkan data penjualan per kategori
@st.cache_data(ttl=300)
def get_category_sales(start_date, end_date):
    query = f"""
    SELECT 
        p.category,
        COUNT(DISTINCT oi.order_id) as num_orders,
        COUNT(DISTINCT oi.user_id) as num_customers,
        SUM(oi.quantity) as total_quantity,
        SUM(oi.sale_price) as total_revenue,
        AVG(oi.sale_price) as avg_order_value
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE DATE(oi.created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND oi.status = 'Complete'
    GROUP BY p.category
    ORDER BY total_revenue DESC
    """
    return run_query(query)

# Fungsi untuk AI dengan caching (VERSI DIPERBAIKI)
@st.cache_data(ttl=3600)  # Cache 1 jam
def get_ai_response(prompt, context=""):
    # Buat hash dari prompt sebagai cache key
    prompt_hash = hashlib.md5(f"{prompt}{context}".encode()).hexdigest()
    
    # Cek di session state
    if prompt_hash in st.session_state.ai_cache:
        return st.session_state.ai_cache[prompt_hash]
    
    # Generate response
    try:
        response = llm.ask(
            prompt=prompt,
            context=context if context else None,
            temperature=0.3,
            use_cache=False  # Cache sudah dihandle oleh st.cache_data
        )
        st.session_state.ai_cache[prompt_hash] = response
        return response
    except Exception as e:
        return f"Error connecting to LLM: {e}"

# ==================== SETUP DATABASE & LLM ====================
# Panggil fungsi inisialisasi
engine = init_connection()
llm = init_llm()

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard Overview", "Sales Analysis", "Customer Insights", "Forecasting", "AI Assistant"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Date Range")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=90))
end_date = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info(
    "Gunakan filter tanggal untuk melihat data dalam periode tertentu. "
    "Dashboard ini menggunakan data real-time dari database lokal."
)

# ==================== HALAMAN 1: DASHBOARD OVERVIEW ====================
if page == "Dashboard Overview":
    st.title("📊 E-commerce Analytics Dashboard")
    st.markdown("### Overview of Key Metrics")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Revenue
    query_revenue = f"""
    SELECT COALESCE(SUM(sale_price), 0) as total_revenue
    FROM order_items
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'Complete'
    """
    revenue_df = run_query(query_revenue)
    revenue = revenue_df['total_revenue'].iloc[0] if not revenue_df.empty else 0
    
    # Total Orders
    query_orders = f"""
    SELECT COALESCE(COUNT(DISTINCT order_id), 0) as total_orders
    FROM order_items
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'Complete'
    """
    orders_df = run_query(query_orders)
    orders = orders_df['total_orders'].iloc[0] if not orders_df.empty else 0
    
    # Average Order Value
    query_aov = f"""
    SELECT COALESCE(AVG(sale_price), 0) as avg_order
    FROM order_items
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'Complete'
    """
    aov_df = run_query(query_aov)
    aov = aov_df['avg_order'].iloc[0] if not aov_df.empty else 0
    
    # Total Customers
    query_customers = f"""
    SELECT COALESCE(COUNT(DISTINCT user_id), 0) as total_customers
    FROM order_items
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'Complete'
    """
    customers_df = run_query(query_customers)
    customers = customers_df['total_customers'].iloc[0] if not customers_df.empty else 0
    
    col1.metric("Total Revenue", f"${revenue:,.0f}")
    col2.metric("Total Orders", f"{orders:,.0f}")
    col3.metric("Avg Order Value", f"${aov:,.2f}")
    col4.metric("Active Customers", f"{customers:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    # Daily Sales Trend
    query_daily = f"""
    SELECT 
        DATE(created_at) as date,
        COALESCE(SUM(sale_price), 0) as daily_sales
    FROM order_items
    WHERE DATE(created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND status = 'Complete'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    df_daily = run_query(query_daily)
    
    if not df_daily.empty:
        fig = px.line(df_daily, x='date', y='daily_sales', title='Daily Sales Trend')
        fig.update_layout(xaxis_title="Date", yaxis_title="Sales ($)")
        col1.plotly_chart(fig, use_container_width=True)
    else:
        col1.info("No data available for selected date range")
    
    # Sales by Category
    query_cat = f"""
    SELECT 
        p.category,
        COALESCE(SUM(oi.sale_price), 0) as total_sales
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE DATE(oi.created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND oi.status = 'Complete'
    GROUP BY p.category
    ORDER BY total_sales DESC
    """
    df_cat = run_query(query_cat)
    
    if not df_cat.empty:
        fig = px.pie(df_cat, values='total_sales', names='category', title='Sales by Category')
        col2.plotly_chart(fig, use_container_width=True)
    else:
        col2.info("No data available for selected date range")
    
    # Recent Orders Table
    st.markdown("### 📋 Recent Orders")
    query_recent = f"""
    SELECT 
        oi.order_id,
        oi.created_at as date,
        u.email,
        p.name as product,
        oi.quantity,
        oi.sale_price
    FROM order_items oi
    JOIN users u ON oi.user_id = u.id
    JOIN products p ON oi.product_id = p.id
    WHERE DATE(oi.created_at) BETWEEN '{start_date}' AND '{end_date}'
    AND oi.status = 'Complete'
    ORDER BY oi.created_at DESC
    LIMIT 10
    """
    df_recent = run_query(query_recent)
    if not df_recent.empty:
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent orders found")

# ==================== HALAMAN 2: SALES ANALYSIS ====================
elif page == "Sales Analysis":
    st.title("📈 Sales Analysis")
    
    # Product Performance
    st.markdown("### 🏆 Top Products")
    df_products = get_product_sales(start_date, end_date)
    
    if not df_products.empty:
        col1, col2 = st.columns(2)
        
        # Top 10 products by revenue
        top_products = df_products.head(10)
        fig = px.bar(top_products, x='product_name', y='total_revenue', 
                     title='Top 10 Products by Revenue',
                     color='category')
        fig.update_layout(xaxis_tickangle=-45)
        col1.plotly_chart(fig, use_container_width=True)
        
        # Top 10 products by quantity
        top_qty = df_products.nlargest(10, 'total_quantity')
        fig = px.bar(top_qty, x='product_name', y='total_quantity',
                     title='Top 10 Products by Quantity Sold',
                     color='category')
        fig.update_layout(xaxis_tickangle=-45)
        col2.plotly_chart(fig, use_container_width=True)
        
        # Product table
        st.markdown("### 📊 Product Performance Details")
        st.dataframe(df_products, use_container_width=True)
        
        # Download button
        csv = df_products.to_csv(index=False)
        st.download_button(
            label="📥 Download Product Data (CSV)",
            data=csv,
            file_name=f"product_sales_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("No product sales data available")
    
    # Category Analysis
    st.markdown("### 📦 Category Analysis")
    df_categories = get_category_sales(start_date, end_date)
    
    if not df_categories.empty:
        col1, col2 = st.columns(2)
        
        fig = px.bar(df_categories, x='category', y='total_revenue',
                     title='Revenue by Category')
        col1.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(df_categories, x='category', y='num_orders',
                     title='Orders by Category')
        col2.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_categories, use_container_width=True)
    else:
        st.info("No category data available")

# ==================== HALAMAN 3: CUSTOMER INSIGHTS ====================
elif page == "Customer Insights":
    st.title("👥 Customer Insights")
    
    # RFM Segments
    st.markdown("### 🎯 Customer Segmentation (RFM Analysis)")
    
    with st.spinner("Loading RFM data..."):
        df_rfm = get_rfm_segments()
    
    if not df_rfm.empty:
        col1, col2 = st.columns([1, 1])
        
        # Distribusi segment
        segment_counts = df_rfm['segment'].value_counts().reset_index()
        segment_counts.columns = ['segment', 'count']
        
        fig = px.pie(segment_counts, values='count', names='segment', 
                     title='Customer Segments Distribution',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        col1.plotly_chart(fig, use_container_width=True)
        
        # RFM metrics per segment
        segment_metrics = df_rfm.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'user_id': 'count'
        }).round(2)
        segment_metrics.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Customer Count']
        segment_metrics = segment_metrics.sort_values('Avg Monetary ($)', ascending=False)
        
        col2.dataframe(segment_metrics, use_container_width=True)
        
        # Segment Insights
        st.markdown("### 💡 Segment Insights")
        
        for segment in segment_metrics.index:
            with st.expander(f"📌 {segment} ({segment_metrics.loc[segment, 'Customer Count']} customers)"):
                metrics = segment_metrics.loc[segment]
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Recency", f"{metrics['Avg Recency (days)']:.1f} days")
                col2.metric("Avg Frequency", f"{metrics['Avg Frequency']:.1f} orders")
                col3.metric("Avg Monetary", f"${metrics['Avg Monetary ($)']:.2f}")
                
                # Recommendation berdasarkan segment
                if segment == "Champions":
                    st.success("🏆 **Recommendation**: Reward them, they can become brand advocates. Offer early access, VIP treatment, and referral programs.")
                elif segment == "Loyal Customers":
                    st.info("💎 **Recommendation**: Increase their loyalty. Offer membership, loyalty programs, and cross-sell higher-value products.")
                elif segment == "Potential Loyalists":
                    st.info("🌟 **Recommendation**: Convert them to loyal. Offer personalized recommendations based on their purchase history.")
                elif segment == "New Customers":
                    st.info("🆕 **Recommendation**: Onboard them properly. Provide excellent customer service and educational content about products.")
                elif segment == "At Risk":
                    st.warning("⚠️ **Recommendation**: Bring them back. Send re-engagement emails with special discounts and new product highlights.")
                elif segment == "Lost":
                    st.error("❌ **Recommendation**: Win them back with aggressive reactivation campaigns, big discounts, and win-back offers.")
                else:
                    st.info("📊 **Recommendation**: Analyze their behavior patterns and consider targeted marketing campaigns.")
        
        # Top customers by segment
        st.markdown("### 🏆 Top Customers by Segment")
        if not df_rfm.empty and 'segment' in df_rfm.columns:
            selected_segment = st.selectbox("Select segment to view top customers", 
                                           df_rfm['segment'].unique())
            
            top_customers = df_rfm[df_rfm['segment'] == selected_segment] \
                            .nlargest(10, 'monetary')[['user_id', 'recency', 'frequency', 'monetary']]
            
            st.dataframe(top_customers, use_container_width=True)
        
        # Download button
        csv = df_rfm.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete RFM Data (CSV)",
            data=csv,
            file_name=f"rfm_segments_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.error("No customer data available. Please check database connection.")

# ==================== HALAMAN 4: FORECASTING ====================
elif page == "Forecasting":
    st.title("📈 Sales Forecasting")
    
    # Pilih model forecasting
    model_type = st.selectbox(
        "Select Forecasting Model",
        ["ARIMA (Default)", "Simple Moving Average", "Exponential Smoothing"]
    )
    
    forecast_days = st.slider("Forecast Period (days)", min_value=7, max_value=90, value=30)
    
    # Load data
    df_sales = get_forecast_data()
    
    if not df_sales.empty and len(df_sales) > 30:
        # Split data
        train_size = int(len(df_sales) * 0.8)
        train = df_sales.iloc[:train_size]
        test = df_sales.iloc[train_size:]
        
        st.markdown(f"### 📊 Forecasting with {model_type}")
        st.info(f"Training data: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
        st.info(f"Testing data: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
        
        # Pilih metrik yang akan di-forecast
        target_metric = st.radio("Select target to forecast", 
                                  ["total_sales", "num_orders", "unique_customers"],
                                  format_func=lambda x: {
                                      "total_sales": "Total Sales ($)",
                                      "num_orders": "Number of Orders",
                                      "unique_customers": "Unique Customers"
                                  }[x])
        
        if model_type == "ARIMA (Default)":
            from statsmodels.tsa.arima.model import ARIMA
            
            with st.spinner("Training ARIMA model..."):
                try:
                    # Simple ARIMA model
                    model = ARIMA(train[target_metric], order=(5,1,2))
                    model_fit = model.fit()
                    
                    # Forecast
                    forecast = model_fit.forecast(steps=forecast_days)
                    forecast_index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), 
                                                    periods=forecast_days)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train.index, y=train[target_metric],
                                            mode='lines', name='Training Data', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=test.index, y=test[target_metric],
                                            mode='lines', name='Actual Test Data', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                                            mode='lines+markers', name='Forecast', 
                                            line=dict(color='red', dash='dash')))
                    
                    fig.update_layout(title=f'{target_metric.replace("_", " ").title()} Forecast',
                                     xaxis_title='Date',
                                     yaxis_title=target_metric.replace("_", " ").title())
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast table
                    st.markdown("### 📋 Forecast Results")
                    forecast_df = pd.DataFrame({
                        'Date': forecast_index,
                        'Forecast': forecast.values
                    })
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Model metrics
                    if len(test) > 0:
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        
                        # Forecast for test period
                        test_forecast = model_fit.forecast(steps=len(test))
                        
                        mae = mean_absolute_error(test[target_metric], test_forecast)
                        rmse = np.sqrt(mean_squared_error(test[target_metric], test_forecast))
                        mape = np.mean(np.abs((test[target_metric] - test_forecast) / test[target_metric])) * 100
                        
                        st.markdown("### 📊 Model Performance")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("MAE", f"{mae:.2f}")
                        col2.metric("RMSE", f"{rmse:.2f}")
                        col3.metric("MAPE", f"{mape:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error in ARIMA model: {e}")
        
        elif model_type == "Simple Moving Average":
            window = st.slider("Moving Average Window (days)", min_value=3, max_value=30, value=7)
            
            # Calculate moving average
            train_ma = train[target_metric].rolling(window=window).mean()
            
            # Simple forecast (last moving average value)
            last_ma = train_ma.iloc[-1]
            forecast = [last_ma] * forecast_days
            forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), 
                                            periods=forecast_days)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train[target_metric],
                                    mode='lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train.index, y=train_ma,
                                    mode='lines', name=f'{window}-day MA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                                    mode='lines+markers', name='Forecast', 
                                    line=dict(color='red', dash='dash')))
            
            fig.update_layout(title=f'Moving Average Forecast (window={window})')
            st.plotly_chart(fig, use_container_width=True)
        
        elif model_type == "Exponential Smoothing":
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            
            alpha = st.slider("Smoothing Level (alpha)", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
            
            with st.spinner("Training Exponential Smoothing model..."):
                try:
                    model = SimpleExpSmoothing(train[target_metric]).fit(smoothing_level=alpha)
                    forecast = model.forecast(forecast_days)
                    forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), 
                                                    periods=forecast_days)
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train.index, y=train[target_metric],
                                            mode='lines', name='Actual', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                                            mode='lines+markers', name='Forecast', 
                                            line=dict(color='red', dash='dash')))
                    
                    fig.update_layout(title=f'Exponential Smoothing Forecast (alpha={alpha})')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in Exponential Smoothing: {e}")
        
        # Download forecast
        if st.button("📥 Download Forecast Data"):
            forecast_df = pd.DataFrame({
                'Date': forecast_index,
                'Forecast': forecast.values if hasattr(forecast, 'values') else forecast
            })
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Confirm Download",
                data=csv,
                file_name=f"forecast_{target_metric}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("Insufficient data for forecasting. Need at least 30 days of data.")

# ==================== HALAMAN 5: AI ASSISTANT ====================
elif page == "AI Assistant":
    st.title("🤖 AI Assistant with Local LLM")
    st.markdown("Ask questions about your data in natural language")
    
    # Tampilkan status LLM
    if llm is None:
        st.error("❌ LLM tidak tersedia. Pastikan Ollama berjalan dan model sudah didownload.")
        st.code("""
        # Di terminal:
        ollama serve
        # Di terminal lain:
        ollama pull gemma2:2b
        """)
    else:
        # Pilihan mode
        mode = st.radio(
            "Select Mode",
            ["General Assistant", "Data Analyst", "SQL Expert"],
            horizontal=True
        )
        
        # Context dari data (untuk mode Data Analyst)
        data_context = {}
        if mode == "Data Analyst":
            with st.spinner("Loading data context..."):
                # Ambil ringkasan data
                summary_query = """
                SELECT 
                    COUNT(DISTINCT user_id) as total_customers,
                    COUNT(DISTINCT order_id) as total_orders,
                    COALESCE(SUM(sale_price), 0) as total_revenue,
                    COALESCE(AVG(sale_price), 0) as avg_order_value,
                    MIN(created_at) as first_order,
                    MAX(created_at) as last_order
                FROM order_items WHERE status = 'Complete'
                """
                df_summary = run_query(summary_query)
                if not df_summary.empty:
                    data_context = {
                        "Total Customers": f"{df_summary['total_customers'].iloc[0]:,}",
                        "Total Orders": f"{df_summary['total_orders'].iloc[0]:,}",
                        "Total Revenue": f"${df_summary['total_revenue'].iloc[0]:,.2f}",
                        "Average Order Value": f"${df_summary['avg_order_value'].iloc[0]:.2f}",
                        "Date Range": f"{pd.to_datetime(df_summary['first_order'].iloc[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(df_summary['last_order'].iloc[0]).strftime('%Y-%m-%d')}"
                    }
        
        tab1, tab2, tab3, tab4 = st.tabs(["Ask Anything", "SQL Generator", "Explain Query", "Data Analysis"])
        
        with tab1:
            user_question = st.text_area("Your question:", height=100, 
                                         placeholder="e.g., What were our best selling products last month?")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                use_context = st.checkbox("Include data context", value=(mode=="Data Analyst"))
            with col2:
                temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            
            if st.button("Ask AI", type="primary"):
                with st.spinner("AI is thinking..."):
                    # Siapkan konteks
                    context = None
                    if use_context and mode == "Data Analyst" and data_context:
                        context = str(data_context)
                    
                    # Kirim pertanyaan
                    response = llm.ask(
                        prompt=user_question,
                        context=context,
                        temperature=temp,
                        use_cache=True
                    )
                    
                    st.markdown("### Response:")
                    st.write(response)
        
        with tab2:
            st.markdown("### Generate SQL from Natural Language")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                complexity = st.selectbox("Query Complexity", ["Simple", "Medium", "Complex"])
            with col2:
                include_examples = st.checkbox("Include few-shot examples", value=True)
            
            nl_query = st.text_area("Describe what you want:", height=100,
                                    placeholder="e.g., Show top 10 customers by total spending including their name and email")
            
            # Few-shot examples
            examples = None
            if include_examples:
                examples = [
                    {
                        "request": "Show total sales by category",
                        "sql": "SELECT p.category, SUM(oi.sale_price) as total_sales FROM order_items oi JOIN products p ON oi.product_id = p.id WHERE oi.status = 'Complete' GROUP BY p.category ORDER BY total_sales DESC;"
                    },
                    {
                        "request": "Find the top 5 customers by number of orders",
                        "sql": "SELECT u.id, u.first_name, u.last_name, COUNT(DISTINCT oi.order_id) as order_count FROM users u JOIN order_items oi ON u.id = oi.user_id WHERE oi.status = 'Complete' GROUP BY u.id, u.first_name, u.last_name ORDER BY order_count DESC LIMIT 5;"
                    }
                ]
            
            if st.button("Generate SQL"):
                with st.spinner("Generating SQL..."):
                    sql = llm.generate_sql(
                        natural_language=nl_query,
                        examples=examples if include_examples else None
                    )
                    
                    st.code(sql, language="sql")
                    
                    if st.button("Run this query"):
                        try:
                            df_result = run_query(sql)
                            if not df_result.empty:
                                st.dataframe(df_result, use_container_width=True)
                                
                                # Download button
                                csv = df_result.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results (CSV)",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("Query returned no results")
                        except Exception as e:
                            st.error(f"Error executing query: {e}")
        
        with tab3:
            st.markdown("### Explain SQL Query")
            sql_query = st.text_area("Paste SQL Query:", height=150)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                detailed = st.checkbox("Detailed explanation", value=False)
            with col2:
                lang = st.radio("Language", ["english", "indonesian"], horizontal=True)
            
            if st.button("Explain"):
                with st.spinner("Analyzing query..."):
                    explanation = llm.explain_sql(
                        sql_query=sql_query,
                        detailed=detailed,
                        language=lang
                    )
                    st.markdown("### Explanation:")
                    st.write(explanation)
        
        with tab4:
            st.markdown("### Data Analysis with AI")
            
            if not data_context:
                st.warning("Load data context first by selecting 'Data Analyst' mode above.")
            else:
                # Tampilkan data context
                with st.expander("Current Data Context", expanded=False):
                    for key, value in data_context.items():
                        st.text(f"{key}: {value}")
                
                analysis_question = st.text_area("What would you like to know about the data?", 
                                                height=100,
                                                placeholder="e.g., What patterns do you see in customer spending?")
                
                include_rec = st.checkbox("Include business recommendations", value=True)
                
                if st.button("Analyze"):
                    with st.spinner("Analyzing data..."):
                        insight = llm.analyze_data(
                            question=analysis_question,
                            df_summary=data_context,
                            include_recommendations=include_rec
                        )
                        st.markdown("### Insights:")
                        st.write(insight)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        E-commerce Analytics Dashboard v2.0 | 100% Local Open Source Stack | 
        <a href='https://github.com/your-repo' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)