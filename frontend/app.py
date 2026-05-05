# frontend/app.py
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "backend"))

from inference_pipeline import InferencePipeline

st.set_page_config(
    page_title="AQI Predictor | Rawalpindi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .aqi-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .model-card {
        padding: 1rem;
        border-radius: 8px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: transform 0.2s;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'daily_summary' not in st.session_state:
    st.session_state.daily_summary = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#10b981"
    elif aqi <= 100:
        return "Moderate", "#f59e0b"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#8b5cf6"
    else:
        return "Hazardous", "#7c3aed"

st.markdown('<div class="main-header">🌫️ AQI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Air Quality Index predictions for Rawalpindi, Pakistan</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    days_ahead = st.slider("Forecast Days", 1, 5, 3)

    st.markdown("### 🤖 Models")
    available_models = ["ensemble", "random_forest", "gradient_boosting", "svr", "knn", "xgboost"]
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Fetching data and generating predictions..."):
            try:
                st.session_state.pipeline = InferencePipeline()
                st.session_state.predictions = st.session_state.pipeline.predict(days_ahead=days_ahead)
                st.session_state.daily_summary = st.session_state.pipeline.get_daily_summary(
                    st.session_state.predictions
                )
                st.success("Predictions generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if st.session_state.predictions is not None:
    predictions = st.session_state.predictions
    daily_summary = st.session_state.daily_summary

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        pred_col = f"{selected_model}_prediction"
        current_aqi = predictions[pred_col].iloc[0] if pred_col in predictions else \
        predictions['ensemble_prediction'].iloc[-1]
        category, color = get_aqi_category(current_aqi)

        st.markdown(f"""
        <div class="aqi-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Current {selected_model.replace('_', ' ').title()} AQI</div>
            <div style="font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0;">{current_aqi:.0f}</div>
            <div style="font-size: 1.1rem; font-weight: 600;">{category}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 Hourly AQI Forecast")

    fig = go.Figure()

    if selected_model == "ensemble":
        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']
        model_names = ['random_forest', 'gradient_boosting', 'svr', 'knn', 'xgboost']

        for i, model in enumerate(model_names):
            col = f"{model}_prediction"
            if col in predictions.columns:
                fig.add_trace(go.Scatter(
                    x=predictions['datetime'],
                    y=predictions[col],
                    mode='lines',
                    name=model.replace('_', ' ').title(),
                    line=dict(color=colors[i], width=1, dash='dot'),
                    opacity=0.5
                ))

        fig.add_trace(go.Scatter(
            x=predictions['datetime'],
            y=predictions['ensemble_prediction'],
            mode='lines',
            name='Ensemble',
            line=dict(color='#1f2937', width=3)
        ))
    else:
        pred_col = f"{selected_model}_prediction"
        fig.add_trace(go.Scatter(
            x=predictions['datetime'],
            y=predictions[pred_col],
            mode='lines',
            name=selected_model.replace('_', ' ').title(),
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))

    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.05, name="Good")
    fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="yellow", opacity=0.05, name="Moderate")
    fig.add_hrect(y0=100, y1=150, line_width=0, fillcolor="orange", opacity=0.05, name="Unhealthy (Sensitive)")
    fig.add_hrect(y0=150, y1=200, line_width=0, fillcolor="red", opacity=0.05, name="Unhealthy")

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="",
        yaxis_title="AQI",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Daily Summary")

    cols = st.columns(len(daily_summary))

    for i, (date, row) in enumerate(daily_summary.iterrows()):
        with cols[i]:
            aqi_value = row[f"{selected_model}_prediction"] if selected_model != "ensemble" else row[
                'ensemble_prediction']
            category, color = get_aqi_category(aqi_value)

            st.markdown(f"""
            <div class="model-card">
                <div class="metric-label">{date.strftime('%a, %b %d')}</div>
                <div class="metric-value" style="color: {color}">{aqi_value:.0f}</div>
                <div style="font-size: 0.8rem; color: {color}">{category}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🤖 Model Comparison")

    model_cols = st.columns(5)
    model_names = ['random_forest', 'gradient_boosting', 'svr', 'knn', 'xgboost']
    colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']

    for i, (model, color) in enumerate(zip(model_names, colors)):
        with model_cols[i]:
            col = f"{model}_prediction"
            if col in predictions.columns:
                avg_aqi = predictions[col].mean()
                category, _ = get_aqi_category(avg_aqi)

                st.markdown(f"""
                <div class="model-card" style="border-left: 3px solid {color}">
                    <div class="metric-label">{model.replace('_', ' ').title()}</div>
                    <div class="metric-value" style="font-size: 1.3rem;">{avg_aqi:.0f}</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">{category}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🌬️</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: #6b7280; margin-bottom: 0.5rem;">
            No Predictions Yet
        </div>
        <div style="color: #9ca3af;">
            Use the sidebar to generate AQI predictions for Rawalpindi
        </div>
    </div>
    """, unsafe_allow_html=True)