import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Sleep Health Analysis",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HP\Desktop\Sleep_health_and_lifestyle_dataset.csv")
    return df

# Preprocess data
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle Blood Pressure - split into systolic and diastolic
    df_processed[['Systolic_BP', 'Diastolic_BP']] = df_processed['Blood Pressure'].str.split('/', expand=True).astype(int)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()
    
    df_processed['Gender_Encoded'] = le_gender.fit_transform(df_processed['Gender'])
    df_processed['Occupation_Encoded'] = le_occupation.fit_transform(df_processed['Occupation'])
    df_processed['BMI_Encoded'] = le_bmi.fit_transform(df_processed['BMI Category'])
    
    # Handle Sleep Disorder (None should be a category)
    df_processed['Sleep Disorder'] = df_processed['Sleep Disorder'].fillna('None')
    
    return df_processed, le_gender, le_occupation, le_bmi

# Main app
def main():
    # Load data
    df = load_data()
    df_processed, le_gender, le_occupation, le_bmi = preprocess_data(df)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", 
                           ["🏠 Home", 
                            "📊 Data Explorer", 
                            "🔍 Deep Dive Analysis",
                            "🎯 Sleep Quality Predictor",
                            "📈 Insights Dashboard"])
    
    # HOME PAGE
    if page == "🏠 Home":
        st.markdown('<p class="main-header">😴 Sleep Health & Lifestyle Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Project Goal
        This project analyzes how **lifestyle factors** (physical activity, stress, diet, screen time) 
        impact **sleep quality** and identifies individuals at risk for sleep disorders.
        
        **Neuroscience Question:** *"How do behavioral patterns affect sleep architecture and cognitive recovery?"*
        """)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Participants", len(df))
        with col2:
            avg_sleep = df['Sleep Duration'].mean()
            st.metric("Avg Sleep Duration", f"{avg_sleep:.1f} hrs")
        with col3:
            avg_quality = df['Quality of Sleep'].mean()
            st.metric("Avg Sleep Quality", f"{avg_quality:.1f}/10")
        with col4:
            disorder_pct = (df['Sleep Disorder'].notna().sum() / len(df)) * 100
            st.metric("With Sleep Disorders", f"{disorder_pct:.1f}%")
        
        st.markdown("---")
        
        # Dataset overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Overview")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Features:** {', '.join(df.columns[:6])}...")
            
            st.subheader("🎭 Demographics")
            gender_counts = df['Gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                        title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💤 Sleep Disorder Distribution")
            disorder_counts = df['Sleep Disorder'].fillna('None').value_counts()
            fig = px.bar(x=disorder_counts.index, y=disorder_counts.values,
                        labels={'x': 'Sleep Disorder', 'y': 'Count'},
                        color=disorder_counts.values,
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Key Statistics")
            st.dataframe(df.describe().round(2))
    
    # DATA EXPLORER PAGE
    elif page == "📊 Data Explorer":
        st.markdown('<p class="main-header">📊 Interactive Data Explorer</p>', unsafe_allow_html=True)
        
        # Filters in sidebar
        st.sidebar.subheader("🔍 Filters")
        
        age_range = st.sidebar.slider("Age Range", 
                                     int(df['Age'].min()), 
                                     int(df['Age'].max()), 
                                     (int(df['Age'].min()), int(df['Age'].max())))
        
        gender_filter = st.sidebar.multiselect("Gender", 
                                               options=df['Gender'].unique().tolist(),
                                               default=df['Gender'].unique().tolist())
        
        occupation_filter = st.sidebar.multiselect("Occupation",
                                                   options=sorted(df['Occupation'].unique().tolist()),
                                                   default=[])
        
        # Apply filters
        filtered_df = df[
            (df['Age'] >= age_range[0]) & 
            (df['Age'] <= age_range[1]) &
            (df['Gender'].isin(gender_filter))
        ]
        
        if occupation_filter:
            filtered_df = filtered_df[filtered_df['Occupation'].isin(occupation_filter)]
        
        st.write(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_sleep_data.csv",
            mime="text/csv"
        )
        
        # Visualizations
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig = px.histogram(filtered_df, x='Age', nbins=20, 
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("BMI Category Distribution")
            bmi_counts = filtered_df['BMI Category'].value_counts()
            fig = px.bar(x=bmi_counts.index, y=bmi_counts.values,
                        color=bmi_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # DEEP DIVE ANALYSIS PAGE
    elif page == "🔍 Deep Dive Analysis":
        st.markdown('<p class="main-header">🔍 Deep Dive Analysis</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Correlations", "📈 Trends", "🎯 Patterns"])
        
        with tab1:
            st.subheader("Correlation Heatmap")
            
            # Select numerical columns for correlation
            numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                            'Physical Activity Level', 'Stress Level', 
                            'Heart Rate', 'Daily Steps']
            
            corr_data = df_processed[numerical_cols + ['Systolic_BP', 'Diastolic_BP']].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            st.markdown("""
            **🔑 Key Correlations:**
            - Strong positive correlation between Sleep Duration and Quality of Sleep
            - Negative correlation between Stress Level and Sleep Quality
            - Physical Activity shows positive relationship with Sleep Quality
            """)
        
        with tab2:
            st.subheader("Sleep Patterns by Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sleep duration by occupation
                occupation_sleep = df.groupby('Occupation')['Sleep Duration'].mean().sort_values(ascending=False)
                fig = px.bar(x=occupation_sleep.values, 
                           y=occupation_sleep.index,
                           orientation='h',
                           labels={'x': 'Average Sleep Duration (hrs)', 'y': 'Occupation'},
                           title='Average Sleep Duration by Occupation')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality of sleep by age group
                df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                        labels=['<30', '30-40', '40-50', '50+'])
                age_quality = df.groupby('Age_Group')['Quality of Sleep'].mean()
                fig = px.line(x=age_quality.index, y=age_quality.values,
                            markers=True,
                            labels={'x': 'Age Group', 'y': 'Avg Quality of Sleep'},
                            title='Sleep Quality by Age Group')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Lifestyle Factors vs Sleep Quality")
            
            # Interactive scatter plot
            x_axis = st.selectbox("X-axis", 
                                 ['Physical Activity Level', 'Stress Level', 
                                  'Daily Steps', 'Heart Rate', 'Age'])
            
            y_axis = st.selectbox("Y-axis", 
                                 ['Quality of Sleep', 'Sleep Duration'],
                                 index=0)
            
            color_by = st.selectbox("Color by", 
                                   ['Sleep Disorder', 'BMI Category', 'Gender'])
            
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                           size='Daily Steps', hover_data=['Occupation', 'Age'],
                           title=f'{y_axis} vs {x_axis}')
            
            st.plotly_chart(fig, use_container_width=True)
    
    # SLEEP QUALITY PREDICTOR PAGE
    elif page == "🎯 Sleep Quality Predictor":
        st.markdown('<p class="main-header">🎯 Sleep Quality Predictor</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Input your lifestyle factors to predict sleep quality risk
        This tool uses patterns from the dataset to estimate your sleep quality.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 80, 35)
            gender = st.selectbox("Gender", ['Male', 'Female'])
            occupation = st.selectbox("Occupation", sorted(df['Occupation'].unique().tolist()))
            bmi_category = st.selectbox("BMI Category", 
                                       ['Normal', 'Normal Weight', 'Overweight', 'Obese'])
        
        with col2:
            sleep_duration = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0, 0.5)
            physical_activity = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
            daily_steps = st.number_input("Daily Steps", 1000, 15000, 7000, 500)
        
        with col3:
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            heart_rate = st.slider("Heart Rate (bpm)", 60, 90, 70)
            systolic_bp = st.number_input("Systolic BP", 90, 150, 120)
            diastolic_bp = st.number_input("Diastolic BP", 60, 100, 80)
        
        if st.button("🔮 Predict Sleep Quality", type="primary"):
            # Simple rule-based prediction (you can replace with ML model later)
            quality_score = 10
            
            # Adjust based on factors
            if sleep_duration < 6:
                quality_score -= 2
            elif sleep_duration > 8:
                quality_score -= 1
            
            if stress_level > 7:
                quality_score -= 2
            elif stress_level > 5:
                quality_score -= 1
            
            if physical_activity < 30:
                quality_score -= 1
            
            if daily_steps < 5000:
                quality_score -= 1
            
            quality_score = max(1, min(10, quality_score))
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Your Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Sleep Quality", f"{quality_score}/10")
            
            with col2:
                risk_level = "Low" if quality_score >= 7 else ("Medium" if quality_score >= 5 else "High")
                st.metric("Risk Level", risk_level)
            
            with col3:
                comparison = df['Quality of Sleep'].mean()
                delta = quality_score - comparison
                st.metric("vs Average", f"{delta:+.1f}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if sleep_duration < 7:
                st.warning("⚠️ Consider increasing sleep duration to 7-9 hours")
            
            if stress_level > 6:
                st.warning("⚠️ High stress detected. Try relaxation techniques")
            
            if physical_activity < 40:
                st.info("ℹ️ Increasing physical activity may improve sleep quality")
            
            if daily_steps < 7000:
                st.info("ℹ️ Aim for at least 7,000-10,000 steps daily")
            
            if quality_score >= 7:
                st.success("✅ Your sleep quality indicators look good! Keep it up!")
    
    # INSIGHTS DASHBOARD PAGE
    elif page == "📈 Insights Dashboard":
        st.markdown('<p class="main-header">📈 Key Insights Dashboard</p>', unsafe_allow_html=True)
        
        # Top insights
        st.subheader("🔑 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>1️⃣ Sleep Duration Impact</h4>
            <p>People sleeping 7-9 hours report significantly higher sleep quality scores.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sleep duration vs quality
            bins = [0, 6, 7, 8, 9, 12]
            labels = ['<6h', '6-7h', '7-8h', '8-9h', '9h+']
            df['Sleep_Category'] = pd.cut(df['Sleep Duration'], bins=bins, labels=labels)
            
            avg_quality = df.groupby('Sleep_Category')['Quality of Sleep'].mean()
            fig = px.bar(x=avg_quality.index, y=avg_quality.values,
                        labels={'x': 'Sleep Duration Category', 'y': 'Avg Quality'},
                        color=avg_quality.values,
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>2️⃣ Stress & Sleep Quality</h4>
            <p>Strong negative correlation between stress levels and sleep quality.</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = px.scatter(df, x='Stress Level', y='Quality of Sleep',
                           trendline='ols',
                           labels={'Stress Level': 'Stress Level (1-10)', 
                                  'Quality of Sleep': 'Sleep Quality (1-10)'},
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>3️⃣ Physical Activity Benefits</h4>
            <p>Higher physical activity levels correlate with better sleep quality.</p>
            </div>
            """, unsafe_allow_html=True)
            
            activity_bins = [0, 40, 60, 80, 100]
            activity_labels = ['Low', 'Medium', 'High', 'Very High']
            df['Activity_Level'] = pd.cut(df['Physical Activity Level'], 
                                         bins=activity_bins, labels=activity_labels)
            
            avg_by_activity = df.groupby('Activity_Level')['Quality of Sleep'].mean()
            fig = px.line(x=avg_by_activity.index, y=avg_by_activity.values,
                         markers=True,
                         labels={'x': 'Physical Activity Level', 'y': 'Avg Sleep Quality'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h4>4️⃣ Sleep Disorders by Occupation</h4>
            <p>Certain occupations show higher prevalence of sleep disorders.</p>
            </div>
            """, unsafe_allow_html=True)
            
            disorder_by_occ = df[df['Sleep Disorder'].notna()].groupby('Occupation').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=disorder_by_occ.values, y=disorder_by_occ.index,
                        orientation='h',
                        labels={'x': 'Count', 'y': 'Occupation'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Summary Statistics")
        
        summary_df = pd.DataFrame({
            'Metric': ['Average Sleep Duration', 'Average Sleep Quality', 
                      'Average Stress Level', 'Average Physical Activity',
                      'Participants with Sleep Disorders'],
            'Value': [
                f"{df['Sleep Duration'].mean():.2f} hours",
                f"{df['Quality of Sleep'].mean():.2f}/10",
                f"{df['Stress Level'].mean():.2f}/10",
                f"{df['Physical Activity Level'].mean():.2f}/100",
                f"{df['Sleep Disorder'].notna().sum()} ({(df['Sleep Disorder'].notna().sum()/len(df)*100):.1f}%)"
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
