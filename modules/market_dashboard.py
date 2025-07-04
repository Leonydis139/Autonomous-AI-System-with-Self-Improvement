import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def render_market_dashboard():
    st.header("📈 Live Market Dashboard")
    preferences = st.session_state.preferences.preferences
    analytics_engine = st.session_state.analytics_engine
    data_provider = st.session_state.data_provider

    col1, col2 = st.columns([2, 1])

    with col1:
        stock_symbol = st.text_input(
            "Stock Symbol", 
            value=preferences['default_stock'], 
            help="Enter stock symbol (e.g., AAPL, GOOGL)"
        )
        period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
        enable_ml = st.checkbox("Enable ML Predictions", value=True)
        if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
            with st.spinner("Analyzing stock data..."):
                analysis = analytics_engine.analyze_stock_trends(stock_symbol, period)
                if "error" not in analysis:
                    # Display metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Current Price", f"${analysis['current_price']:.2f}")
                    with col_b:
                        st.metric("Trend", analysis['trend'])
                    with col_c:
                        st.metric("RSI", f"{analysis['rsi']:.2f}")
                    with col_d:
                        st.metric("Volatility", f"{analysis['volatility']:.2f}")

                    # Interactive chart
                    data = analysis['data']
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=[f"{stock_symbol} Price Chart", "Volume"],
                        vertical_spacing=0.1,
                        shared_xaxes=True
                    )
                    fig.add_trace(
                        go.Candlestick(
                            x=data['Date'],
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )
                    if 'MA_5' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data['Date'],
                                y=data['MA_5'],
                                mode='lines',
                                name='MA 5',
                                line=dict(color='orange', width=2)
                            ),
                            row=1, col=1
                        )
                    if 'MA_20' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data['Date'],
                                y=data['MA_20'],
                                mode='lines',
                                name='MA 20',
                                line=dict(color='red', width=2)
                            ),
                            row=1, col=1
                        )
                    fig.add_trace(
                        go.Bar(
                            x=data['Date'],
                            y=data['Volume'],
                            name="Volume",
                            marker_color='#1f77b4'
                        ),
                        row=2, col=1
                    )
                    fig.update_layout(
                        title=f"{stock_symbol} Stock Analysis",
                        height=600,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # ML Prediction
                    if enable_ml and analysis.get('prediction'):
                        prediction = analysis['prediction']
                        if "error" not in prediction:
                            st.subheader("🤖 ML Prediction")
                            col_pred1, col_pred2, col_pred3 = st.columns(3)
                            with col_pred1:
                                st.metric("Predicted Price", f"${prediction['next_price']:.2f}")
                            with col_pred2:
                                st.metric("R² Score", f"{prediction['r2_score']:.3f}")
                            with col_pred3:
                                st.metric("Confidence", prediction['confidence'])
                            if prediction['confidence'] != "High":
                                st.warning("⚠️ Prediction confidence is not high. Use with caution.")
                else:
                    st.error(f"Error: {analysis['error']}")

    with col2:
        st.subheader("💰 Cryptocurrency")
        crypto_coins = list(data_provider.crypto_ids.keys())
        for coin in crypto_coins[:5]:
            crypto_data = data_provider.get_crypto_data(coin)
            if crypto_data and 'price' in crypto_data:
                price = crypto_data['price']
                change = crypto_data.get('change_24h', 0)
                st.metric(
                    coin.capitalize(),
                    f"${price:,.2f}",
                    f"{change:.2f}%" if change else None,
                    delta_color="inverse"
                )
        st.markdown("---")
        st.subheader("📊 Economic Indicators")
        eco_data = data_provider.get_economic_indicators()
        for indicator, data in eco_data.items():
            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                st.metric(
                    indicator,
                    latest.get('value', 'N/A'),
                    data['metadata'].get('unit', '')
                )
