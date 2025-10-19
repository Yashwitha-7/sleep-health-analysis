import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sleep Health Analysis",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css()

# Set default plotly template to dark
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/sleep_health_cleaned.csv')
    with open('data/summary_statistics.json', 'r') as f:
        summary_stats = json.load(f)
    return df, summary_stats

df, summary_stats = load_data()

# Sidebar navigation info
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Use the pages in the sidebar to explore:")
    st.markdown("- **Data Explorer**: Filter and examine the dataset")
    st.markdown("- **Deep Dive Analysis**: Comprehensive statistical insights")
    st.markdown("- **Sleep Quality Predictor**: Personalized risk assessment")
    st.markdown("- **Insights Dashboard**: Key findings and recommendations")
    
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("This application helps you understand how lifestyle factors impact sleep quality and identify potential sleep disorder risks.")
    
    # Cute cat animation
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <div style='font-size: 4rem; animation: float 3s ease-in-out infinite;'>
            🐱
        </div>
        <p style='color: #A0AEC0; font-size: 0.9rem;'>Your sleep companion</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.title("Sleep Health & Lifestyle Analysis")
st.markdown("### Understanding the Connection Between Daily Habits and Sleep Quality")

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 10px; color: white; margin: 2rem 0;'>
    <h3 style='color: white !important; margin: 0; padding-bottom: 0.5rem;'>Welcome to Your Sleep Health Dashboard</h3>
    <p style='margin: 0; font-size: 1.1rem; color: white !important;'>
        Explore comprehensive insights into how lifestyle choices affect sleep quality. 
        This tool analyzes data from 374 individuals to help you understand patterns and make informed decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Key metrics dashboard
st.markdown("## Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Participants",
        value=f"{summary_stats['dataset_info']['total_records']}"
    )

with col2:
    st.metric(
        label="Average Sleep Duration",
        value=f"{summary_stats['sleep_metrics']['avg_sleep_duration']:.1f} hrs"
    )

with col3:
    st.metric(
        label="Average Sleep Quality",
        value=f"{summary_stats['sleep_metrics']['avg_sleep_quality']:.1f}/10"
    )

with col4:
    st.metric(
        label="Sleep Disorder Rate",
        value=f"{summary_stats['sleep_metrics']['disorder_prevalence']:.1f}%"
    )

st.markdown("---")

# Interactive summary visualizations
st.markdown("## Quick Insights")

tab1, tab2, tab3 = st.tabs(["Sleep Quality Distribution", "Demographic Overview", "Health Indicators"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep duration distribution
        fig_duration = px.histogram(
            df, 
            x='Sleep Duration',
            nbins=20,
            title='Sleep Duration Distribution',
            color_discrete_sequence=['#3498DB']
        )
        fig_duration.update_layout(
            xaxis_title='Hours of Sleep',
            yaxis_title='Number of Participants',
            plot_bgcolor='#1A1D24',
            paper_bgcolor='#1A1D24',
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
            xaxis=dict(gridcolor='#2D3748'),
            yaxis=dict(gridcolor='#2D3748')
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Sleep quality distribution
        fig_quality = px.histogram(
            df,
            x='Quality of Sleep',
            nbins=10,
            title='Sleep Quality Scores',
            color_discrete_sequence=['#2ECC71']
        )
        fig_quality.update_layout(
            xaxis_title='Quality Score (1-10)',
            yaxis_title='Number of Participants',
            plot_bgcolor='#1A1D24',
            paper_bgcolor='#1A1D24',
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
            xaxis=dict(gridcolor='#2D3748'),
            yaxis=dict(gridcolor='#2D3748')
        )
        st.plotly_chart(fig_quality, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title='Gender Distribution',
            color_discrete_sequence=['#3498DB', '#E74C3C']
        )
        fig_gender.update_layout(
            paper_bgcolor='#1A1D24',
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8')
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Age distribution
        fig_age = px.box(
            df,
            y='Age',
            title='Age Distribution',
            color_discrete_sequence=['#9B59B6']
        )
        fig_age.update_layout(
            yaxis_title='Age (years)',
            plot_bgcolor='#1A1D24',
            paper_bgcolor='#1A1D24',
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
            xaxis=dict(gridcolor='#2D3748'),
            yaxis=dict(gridcolor='#2D3748')
        )
        st.plotly_chart(fig_age, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep disorder distribution
        disorder_counts = df['Sleep Disorder'].value_counts()
        fig_disorder = px.bar(
            x=disorder_counts.index,
            y=disorder_counts.values,
            title='Sleep Disorder Distribution',
            color=disorder_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Sleep Disorder', 'y': 'Count'}
        )
        fig_disorder.update_layout(
            plot_bgcolor='#1A1D24',
            paper_bgcolor='#1A1D24',
            showlegend=False,
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
            xaxis=dict(gridcolor='#2D3748'),
            yaxis=dict(gridcolor='#2D3748')
        )
        st.plotly_chart(fig_disorder, use_container_width=True)
    
    with col2:
        # BMI categories
        bmi_counts = df['BMI Category'].value_counts()
        fig_bmi = px.bar(
            x=bmi_counts.index,
            y=bmi_counts.values,
            title='BMI Category Distribution',
            color=bmi_counts.values,
            color_continuous_scale='Greens',
            labels={'x': 'BMI Category', 'y': 'Count'}
        )
        fig_bmi.update_layout(
            plot_bgcolor='#1A1D24',
            paper_bgcolor='#1A1D24',
            showlegend=False,
            font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
            xaxis=dict(gridcolor='#2D3748'),
            yaxis=dict(gridcolor='#2D3748')
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

st.markdown("---")

# Key findings section
st.markdown("## Key Findings at a Glance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 1.5rem; border-radius: 10px; color: white;'>
        <h4 style='color: white !important; margin: 0;'>Physical Activity Matters</h4>
        <p style='margin: 0.5rem 0 0 0; color: white !important;'>
            Higher physical activity levels correlate with better sleep quality scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 1.5rem; border-radius: 10px; color: white;'>
        <h4 style='color: white !important; margin: 0;'>Stress Impact</h4>
        <p style='margin: 0.5rem 0 0 0; color: white !important;'>
            Elevated stress levels show strong negative correlation with sleep quality.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; color: white;'>
        <h4 style='color: white !important; margin: 0;'>Occupation Differences</h4>
        <p style='margin: 0.5rem 0 0 0; color: white !important;'>
            Sleep patterns vary significantly across different professions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Call to action
st.markdown("## Ready to Explore?")
st.markdown("""
Use the navigation menu on the left to:
- **Explore the data** with interactive filters
- **Dive deep** into statistical relationships
- **Assess your own** sleep quality risk
- **Discover actionable insights** for better sleep

Navigate to any section using the sidebar menu to begin your analysis.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #A0AEC0; padding: 2rem;'>
    <p>Sleep Health & Lifestyle Analysis Dashboard</p>
    <p style='font-size: 0.9rem;'>Analyzing lifestyle factors and their impact on sleep quality</p>
</div>
""", unsafe_allow_html=True)