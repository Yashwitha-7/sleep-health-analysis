import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

# Load custom CSS
def load_css():
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except:
    pass

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/sleep_health_cleaned.csv')

df = load_data()

# Page header
st.title("Data Explorer")
st.markdown("### Interactive Dataset Exploration and Filtering")

st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; 
            border-left: 4px solid #3498DB; margin-bottom: 2rem;'>
    <p style='margin: 0; color: #2C3E50;'>
        Use the filters below to explore specific subsets of the data. 
        All visualizations and statistics update dynamically based on your selections.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("## Filter Options")
st.sidebar.markdown("Refine the dataset by selecting criteria below:")

# Age filter
age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

# Gender filter
gender_options = ['All'] + list(df['Gender'].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

# Occupation filter
occupation_options = ['All'] + sorted(df['Occupation'].unique().tolist())
selected_occupation = st.sidebar.multiselect(
    "Occupation",
    occupation_options,
    default=['All']
)

# Sleep disorder filter
disorder_options = ['All'] + list(df['Sleep Disorder'].unique())
selected_disorder = st.sidebar.selectbox("Sleep Disorder", disorder_options)

# BMI Category filter
bmi_options = ['All'] + list(df['BMI Category'].unique())
selected_bmi = st.sidebar.selectbox("BMI Category", bmi_options)

# Sleep duration filter
sleep_duration_range = st.sidebar.slider(
    "Sleep Duration (hours)",
    min_value=float(df['Sleep Duration'].min()),
    max_value=float(df['Sleep Duration'].max()),
    value=(float(df['Sleep Duration'].min()), float(df['Sleep Duration'].max())),
    step=0.5
)

# Quality of sleep filter
quality_range = st.sidebar.slider(
    "Quality of Sleep",
    min_value=int(df['Quality of Sleep'].min()),
    max_value=int(df['Quality of Sleep'].max()),
    value=(int(df['Quality of Sleep'].min()), int(df['Quality of Sleep'].max()))
)

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range[0]) & 
    (filtered_df['Age'] <= age_range[1])
]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if 'All' not in selected_occupation:
    filtered_df = filtered_df[filtered_df['Occupation'].isin(selected_occupation)]

if selected_disorder != 'All':
    filtered_df = filtered_df[filtered_df['Sleep Disorder'] == selected_disorder]

if selected_bmi != 'All':
    filtered_df = filtered_df[filtered_df['BMI Category'] == selected_bmi]

filtered_df = filtered_df[
    (filtered_df['Sleep Duration'] >= sleep_duration_range[0]) &
    (filtered_df['Sleep Duration'] <= sleep_duration_range[1])
]

filtered_df = filtered_df[
    (filtered_df['Quality of Sleep'] >= quality_range[0]) &
    (filtered_df['Quality of Sleep'] <= quality_range[1])
]

# Display filtered results count
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filtered Records", len(filtered_df))
with col2:
    st.metric("Percentage of Total", f"{len(filtered_df)/len(df)*100:.1f}%")
with col3:
    if len(filtered_df) > 0:
        st.metric("Avg Sleep Quality", f"{filtered_df['Quality of Sleep'].mean():.1f}/10")
    else:
        st.metric("Avg Sleep Quality", "N/A")

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Data Table", "Summary Statistics", "Visual Overview"])

with tab1:
    st.markdown("## Filtered Dataset")
    
    if len(filtered_df) > 0:
        # Column selector
        all_columns = filtered_df.columns.tolist()
        default_columns = ['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 
                          'Quality of Sleep', 'Stress Level', 'Sleep Disorder']
        
        selected_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=default_columns
        )
        
        if selected_columns:
            st.dataframe(
                filtered_df[selected_columns].style.background_gradient(
                    subset=['Quality of Sleep'] if 'Quality of Sleep' in selected_columns else [],
                    cmap='RdYlGn'
                ),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df[selected_columns].to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_sleep_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one column to display.")
    else:
        st.warning("No records match the selected filters. Please adjust your criteria.")

with tab2:
    st.markdown("## Statistical Summary")
    
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Numerical Features")
            numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                            'Physical Activity Level', 'Stress Level', 
                            'Heart Rate', 'Daily Steps']
            st.dataframe(
                filtered_df[numerical_cols].describe().round(2),
                use_container_width=True
            )
        
        with col2:
            st.markdown("### Categorical Features")
            
            categorical_summary = pd.DataFrame({
                'Feature': ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'],
                'Unique Values': [
                    filtered_df['Gender'].nunique(),
                    filtered_df['Occupation'].nunique(),
                    filtered_df['BMI Category'].nunique(),
                    filtered_df['Sleep Disorder'].nunique()
                ],
                'Most Common': [
                    filtered_df['Gender'].mode()[0] if len(filtered_df['Gender'].mode()) > 0 else 'N/A',
                    filtered_df['Occupation'].mode()[0] if len(filtered_df['Occupation'].mode()) > 0 else 'N/A',
                    filtered_df['BMI Category'].mode()[0] if len(filtered_df['BMI Category'].mode()) > 0 else 'N/A',
                    filtered_df['Sleep Disorder'].mode()[0] if len(filtered_df['Sleep Disorder'].mode()) > 0 else 'N/A'
                ]
            })
            st.dataframe(categorical_summary, use_container_width=True)
        
        st.markdown("### Correlation with Sleep Quality")
        correlations = filtered_df[numerical_cols].corr()['Quality of Sleep'].sort_values(ascending=False)
        correlations = correlations[correlations.index != 'Quality of Sleep']
        
        fig_corr = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title='Feature Correlations with Sleep Quality',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
            color=correlations.values,
            color_continuous_scale='RdBu',
            range_color=[-1, 1]
        )
        fig_corr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Helvetica Neue", size=12, color='#2C3E50'),
            height=400
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No records match the selected filters.")

with tab3:
    st.markdown("## Visual Overview")
    
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep duration vs quality scatter
            fig_scatter = px.scatter(
                filtered_df,
                x='Sleep Duration',
                y='Quality of Sleep',
                color='Sleep Disorder',
                size='Physical Activity Level',
                hover_data=['Age', 'Occupation', 'Stress Level'],
                title='Sleep Duration vs Quality',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_scatter.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Helvetica Neue", size=12, color='#2C3E50')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Stress vs quality scatter
            fig_stress = px.scatter(
                filtered_df,
                x='Stress Level',
                y='Quality of Sleep',
                color='BMI Category',
                size='Daily Steps',
                hover_data=['Age', 'Occupation', 'Sleep Duration'],
                title='Stress Level vs Sleep Quality',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_stress.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Helvetica Neue", size=12, color='#2C3E50')
            )
            st.plotly_chart(fig_stress, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot for sleep quality by occupation
            fig_box = px.box(
                filtered_df,
                x='Occupation',
                y='Quality of Sleep',
                title='Sleep Quality Distribution by Occupation',
                color='Occupation',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_box.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Helvetica Neue", size=12, color='#2C3E50'),
                showlegend=False
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Violin plot for physical activity
            fig_violin = px.violin(
                filtered_df,
                x='Sleep Disorder',
                y='Physical Activity Level',
                title='Physical Activity Distribution by Sleep Disorder',
                color='Sleep Disorder',
                box=True,
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_violin.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Helvetica Neue", size=12, color='#2C3E50'),
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.warning("No records match the selected filters.")

# Cute cat footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <span style='font-size: 2rem;'>🐱</span>
    <p style='color: #7F8C8D; font-size: 0.9rem; margin-top: 0.5rem;'>
        Exploring data with curiosity
    </p>
</div>
""", unsafe_allow_html=True)