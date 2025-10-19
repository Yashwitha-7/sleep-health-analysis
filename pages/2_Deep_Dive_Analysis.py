import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

# ---------------------------
# Page + Theme Configuration
# ---------------------------
st.set_page_config(page_title="Deep Dive Analysis", page_icon="📊", layout="wide")

# Load custom CSS
def load_css():
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css()

# Plotly dark template
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Consistent dark layout helper
def get_plotly_layout():
    return dict(
        plot_bgcolor='#1A1D24',
        paper_bgcolor='#1A1D24',
        font=dict(family="Helvetica Neue", size=12, color='#E8E8E8'),
        xaxis=dict(gridcolor='#2D3748', zerolinecolor='#2D3748'),
        yaxis=dict(gridcolor='#2D3748', zerolinecolor='#2D3748')
    )

# -----------
# Load Data
# -----------
@st.cache_data
def load_data():
    return pd.read_csv('data/sleep_health_cleaned.csv')

df = load_data()

# Extract BP components if not already present
if 'Systolic_BP' not in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# -----------
# Header
# -----------
st.title("Deep Dive Analysis")
st.markdown("### Comprehensive Statistical Insights and Correlations")

# Gradient info box (white text enforced for dark mode)
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
    <p style='margin: 0; color: white !important;'>
        Explore detailed statistical relationships, demographic trends, and health indicators
        that influence sleep quality. This section provides comprehensive analytical views.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------
# Controls
# -----------
analysis_type = st.radio(
    "Select Analysis Type",
    ["Correlation Analysis", "Demographic Trends", "Occupation Analysis",
     "Health Indicators", "Sleep Disorder Profiles"],
    horizontal=True
)

st.markdown("---")

# ----------------------
# Correlation Analysis
# ----------------------
if analysis_type == "Correlation Analysis":
    st.markdown("## Correlation Analysis")
    st.markdown("Understanding relationships between variables")

    numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                      'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']

    # Comprehensive correlation matrix
    st.markdown("### Comprehensive Correlation Matrix")
    corr_matrix = df[numerical_cols].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig_heatmap.update_layout(
        title='Correlation Heatmap of All Numerical Features',
        width=900,
        height=700,
        **get_plotly_layout()
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Key correlations with sleep quality
    st.markdown("### Key Correlations with Sleep Quality")
    sleep_quality_corr = corr_matrix['Quality of Sleep'].drop('Quality of Sleep').sort_values()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Positive Correlations")
        positive_corr = sleep_quality_corr[sleep_quality_corr > 0].sort_values(ascending=False)
        for feature, corr_value in positive_corr.items():
            st.markdown(f"""
            <div style='background-color: #1A3A30; padding: 0.8rem;
                        border-radius: 5px; margin-bottom: 0.5rem; border-left: 4px solid #2ECC71; color: white !important;'>
                <strong>{feature}</strong>: {corr_value:.3f}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Negative Correlations")
        negative_corr = sleep_quality_corr[sleep_quality_corr < 0].sort_values()
        for feature, corr_value in negative_corr.items():
            st.markdown(f"""
            <div style='background-color: #3A1E1E; padding: 0.8rem;
                        border-radius: 5px; margin-bottom: 0.5rem; border-left: 4px solid #E74C3C; color: white !important;'>
                <strong>{feature}</strong>: {corr_value:.3f}
            </div>
            """, unsafe_allow_html=True)

    # Scatter matrix for selected features
    st.markdown("### Relationships Between Key Variables")
    key_features = st.multiselect(
        "Select features to compare (choose 2-4 features)",
        numerical_cols,
        default=['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
    )

    if len(key_features) >= 2:
        fig_scatter_matrix = px.scatter_matrix(
            df,
            dimensions=key_features,
            color='Sleep Disorder',
            title='Scatter Matrix of Selected Features',
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=700
        )
        fig_scatter_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
        fig_scatter_matrix.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)

# ----------------------
# Demographic Trends
# ----------------------
elif analysis_type == "Demographic Trends":
    st.markdown("## Demographic Trends")
    st.markdown("Analyzing patterns across age groups and gender")

    df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70],
                             labels=['20-30', '30-40', '40-50', '50-60', '60+'])

    col1, col2 = st.columns(2)
    with col1:
        age_stats = df.groupby('Age_Group').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean'
        }).reset_index()

        fig_age = go.Figure()
        fig_age.add_trace(go.Scatter(
            x=age_stats['Age_Group'],
            y=age_stats['Sleep Duration'],
            mode='lines+markers',
            name='Sleep Duration',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=10)
        ))
        fig_age.add_trace(go.Scatter(
            x=age_stats['Age_Group'],
            y=age_stats['Quality of Sleep'],
            mode='lines+markers',
            name='Sleep Quality',
            line=dict(color='#2ECC71', width=3),
            marker=dict(size=10)
        ))
        fig_age.update_layout(
            title='Sleep Metrics Across Age Groups',
            xaxis_title='Age Group',
            yaxis_title='Value',
            hovermode='x unified',
            **get_plotly_layout()
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        gender_stats = df.groupby('Gender').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean'
        }).reset_index()

        fig_gender = go.Figure()
        metrics = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
        colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
        for metric, color in zip(metrics, colors):
            fig_gender.add_trace(go.Bar(
                name=metric,
                x=gender_stats['Gender'],
                y=gender_stats[metric],
                marker_color=color
            ))
        fig_gender.update_layout(
            title='Comparison by Gender',
            barmode='group',
            xaxis_title='Gender',
            yaxis_title='Average Value',
            **get_plotly_layout()
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # Statistical testing
    st.markdown("### Statistical Significance Testing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Gender Differences in Sleep Quality")
        male_quality = df[df['Gender'] == 'Male']['Quality of Sleep']
        female_quality = df[df['Gender'] == 'Female']['Quality of Sleep']
        t_stat, p_value = stats.ttest_ind(male_quality, female_quality)
        st.markdown(f"""
        <div style='background-color: #1A1D24; padding: 1rem; border-radius: 8px;
                    border: 2px solid #3498DB; color: #E8E8E8;'>
            <p style='margin: 0;'><strong>T-statistic:</strong> {t_stat:.4f}</p>
            <p style='margin: 0.5rem 0 0 0;'><strong>P-value:</strong> {p_value:.4f}</p>
            <p style='margin: 0.5rem 0 0 0;'><strong>Result:</strong>
                {'Statistically significant difference' if p_value < 0.05 else 'No significant difference'}
                at 0.05 level
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Age Group Analysis (ANOVA)")
        age_groups_data = [df[df['Age_Group'] == group]['Quality of Sleep'].values
                           for group in df['Age_Group'].unique() if pd.notna(group)]
        f_stat, p_value_anova = stats.f_oneway(*age_groups_data)
        st.markdown(f"""
        <div style='background-color: #1A1D24; padding: 1rem; border-radius: 8px;
                    border: 2px solid #2ECC71; color: #E8E8E8;'>
            <p style='margin: 0;'><strong>F-statistic:</strong> {f_stat:.4f}</p>
            <p style='margin: 0.5rem 0 0 0;'><strong>P-value:</strong> {p_value_anova:.4f}</p>
            <p style='margin: 0.5rem 0 0 0;'><strong>Result:</strong>
                {'Significant differences across age groups' if p_value_anova < 0.05 else 'No significant differences'}
                at 0.05 level
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------
# Occupation Analysis
# ----------------------
elif analysis_type == "Occupation Analysis":
    st.markdown("## Occupation Analysis")
    st.markdown("Sleep patterns and health metrics across different professions")

    occupation_stats = df.groupby('Occupation').agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Stress Level': 'mean',
        'Physical Activity Level': 'mean',
        'Person ID': 'count'
    }).round(2)
    occupation_stats.columns = ['Avg Sleep Duration', 'Avg Sleep Quality',
                                'Avg Stress', 'Avg Activity', 'Count']
    occupation_stats = occupation_stats.sort_values('Avg Sleep Quality', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Best Sleep Quality by Occupation")
        top_5 = occupation_stats.nlargest(5, 'Avg Sleep Quality')
        fig_top = px.bar(
            top_5.reset_index(),
            x='Avg Sleep Quality',
            y='Occupation',
            orientation='h',
            title='Top 5 Occupations by Sleep Quality',
            color='Avg Sleep Quality',
            color_continuous_scale='Greens',
            text='Avg Sleep Quality'
        )
        fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_top.update_layout(**get_plotly_layout(), showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown("### Highest Stress by Occupation")
        high_stress = occupation_stats.nlargest(5, 'Avg Stress')
        fig_stress = px.bar(
            high_stress.reset_index(),
            x='Avg Stress',
            y='Occupation',
            orientation='h',
            title='Top 5 Occupations by Stress Level',
            color='Avg Stress',
            color_continuous_scale='Reds',
            text='Avg Stress'
        )
        fig_stress.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_stress.update_layout(**get_plotly_layout(), showlegend=False)
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("### Comprehensive Occupation Comparison")
    fig_occupation = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sleep Duration', 'Sleep Quality', 'Stress Level', 'Physical Activity'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    occupations = occupation_stats.index.tolist()

    fig_occupation.add_trace(
        go.Bar(x=occupations, y=occupation_stats['Avg Sleep Duration'],
               marker_color='#3498DB', name='Sleep Duration'),
        row=1, col=1
    )
    fig_occupation.add_trace(
        go.Bar(x=occupations, y=occupation_stats['Avg Sleep Quality'],
               marker_color='#2ECC71', name='Sleep Quality'),
        row=1, col=2
    )
    fig_occupation.add_trace(
        go.Bar(x=occupations, y=occupation_stats['Avg Stress'],
               marker_color='#E74C3C', name='Stress Level'),
        row=2, col=1
    )
    fig_occupation.add_trace(
        go.Bar(x=occupations, y=occupation_stats['Avg Activity'],
               marker_color='#F39C12', name='Physical Activity'),
        row=2, col=2
    )
    fig_occupation.update_xaxes(tickangle=45)
    fig_occupation.update_layout(height=700, showlegend=False, **get_plotly_layout())
    st.plotly_chart(fig_occupation, use_container_width=True)

    st.markdown("### Detailed Occupation Statistics")
    st.dataframe(
        occupation_stats.style.background_gradient(cmap='RdYlGn', subset=['Avg Sleep Quality']),
        use_container_width=True
    )

# ----------------------
# Health Indicators
# ----------------------
elif analysis_type == "Health Indicators":
    st.markdown("## Health Indicators")
    st.markdown("Cardiovascular and physical health metrics analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Heart Rate", f"{df['Heart Rate'].mean():.1f} BPM")
    with col2:
        st.metric("Avg Systolic BP", f"{df['Systolic_BP'].mean():.1f} mmHg")
    with col3:
        st.metric("Avg Diastolic BP", f"{df['Diastolic_BP'].mean():.1f} mmHg")
    with col4:
        st.metric("Avg Daily Steps", f"{df['Daily Steps'].mean():.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### BMI Category Impact on Sleep")
        bmi_sleep = df.groupby('BMI Category').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Heart Rate': 'mean'
        }).reset_index()

        fig_bmi = go.Figure()
        fig_bmi.add_trace(go.Bar(
            name='Sleep Duration',
            x=bmi_sleep['BMI Category'],
            y=bmi_sleep['Sleep Duration'],
            marker_color='#3498DB'
        ))
        fig_bmi.add_trace(go.Bar(
            name='Sleep Quality',
            x=bmi_sleep['BMI Category'],
            y=bmi_sleep['Quality of Sleep'],
            marker_color='#2ECC71'
        ))
        fig_bmi.update_layout(
            barmode='group',
            title='Sleep Metrics by BMI Category',
            xaxis_title='BMI Category',
            yaxis_title='Average Value',
            **get_plotly_layout()
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    with col2:
        st.markdown("### Heart Rate Distribution by Sleep Quality")
        fig_hr = px.box(
            df.assign(QoS_Group=lambda d: d['Quality of Sleep'].round().astype(int).astype(str)),
            x='QoS_Group',
            y='Heart Rate',
            color='QoS_Group',
            title='Heart Rate vs Sleep Quality',
            color_discrete_sequence=px.colors.sequential.Viridis
)
        fig_hr.update_layout(**get_plotly_layout(), showlegend=False, xaxis_title='Quality of Sleep (rounded)')
        st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("### Blood Pressure Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_bp_scatter = px.scatter(
            df,
            x='Systolic_BP',
            y='Diastolic_BP',
            color='Quality of Sleep',
            size='Heart Rate',
            hover_data=['Age', 'Sleep Duration', 'Stress Level'],
            title='Blood Pressure Relationship',
            color_continuous_scale='RdYlGn'
        )
        fig_bp_scatter.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_bp_scatter, use_container_width=True)

    with col2:
        fig_activity = px.scatter(
            df,
            x='Physical Activity Level',
            y='Quality of Sleep',
            color='BMI Category',
            size='Daily Steps',
            hover_data=['Age', 'Occupation', 'Stress Level'],
            title='Physical Activity vs Sleep Quality',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_activity.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_activity, use_container_width=True)

    st.markdown("### Daily Steps and Sleep Quality Relationship")
    df['Steps_Category'] = pd.cut(df['Daily Steps'],
                                  bins=[0, 5000, 7500, 10000, 15000],
                                  labels=['<5k', '5k-7.5k', '7.5k-10k', '>10k'])
    steps_quality = df.groupby('Steps_Category')['Quality of Sleep'].mean().reset_index()

    fig_steps = px.line(
        steps_quality,
        x='Steps_Category',
        y='Quality of Sleep',
        title='Average Sleep Quality by Daily Steps Category',
        markers=True,
        line_shape='spline'
    )
    fig_steps.update_traces(line_color='#3498DB', marker=dict(size=12, color='#E74C3C'))
    fig_steps.update_layout(
        xaxis_title='Daily Steps Category',
        yaxis_title='Average Sleep Quality',
        **get_plotly_layout()
    )
    st.plotly_chart(fig_steps, use_container_width=True)

# ----------------------
# Sleep Disorder Profiles
# ----------------------
elif analysis_type == "Sleep Disorder Profiles":
    st.markdown("## Sleep Disorder Profiles")
    st.markdown("Comprehensive analysis of different sleep disorder characteristics")

    disorder_counts = df['Sleep Disorder'].value_counts()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Distribution")
        fig_disorder_pie = px.pie(
            values=disorder_counts.values,
            names=disorder_counts.index,
            title='Sleep Disorder Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_disorder_pie.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_disorder_pie, use_container_width=True)

        st.markdown("### Prevalence Statistics")
        for disorder, count in disorder_counts.items():
            percentage = (count / len(df)) * 100
            st.markdown(f"""
            <div style='background-color: #1A1D24; padding: 0.8rem;
                        border-radius: 5px; margin-bottom: 0.5rem;
                        border-left: 4px solid #3498DB; color: #E8E8E8;'>
                <strong>{disorder}</strong>: {count} ({percentage:.1f}%)
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Characteristic Comparison")
        disorder_profile = df.groupby('Sleep Disorder').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean',
            'Heart Rate': 'mean',
            'Age': 'mean'
        }).round(2)

        disorders = disorder_profile.index.tolist()
        fig_radar = go.Figure()
        categories = ['Sleep Duration', 'Sleep Quality', 'Activity Level',
                      'Stress Level (inverted)', 'Heart Rate']

        for disorder in disorders:
            values = [
                disorder_profile.loc[disorder, 'Sleep Duration'],
                disorder_profile.loc[disorder, 'Quality of Sleep'],
                disorder_profile.loc[disorder, 'Physical Activity Level'],
                10 - disorder_profile.loc[disorder, 'Stress Level'],  # invert stress
                disorder_profile.loc[disorder, 'Heart Rate'] / 10
            ]
            values += values[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=disorder
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], gridcolor='#2D3748'),
                angularaxis=dict(gridcolor='#2D3748')
            ),
            showlegend=True,
            title='Sleep Disorder Profiles (Normalized)',
            height=500,
            **get_plotly_layout()
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### Detailed Metric Comparisons")
    col1, col2 = st.columns(2)
    with col1:
        fig_duration = px.box(
            df,
            x='Sleep Disorder',
            y='Sleep Duration',
            title='Sleep Duration Distribution by Disorder',
            color='Sleep Disorder',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_duration.update_layout(**get_plotly_layout(), showlegend=False)
        st.plotly_chart(fig_duration, use_container_width=True)

    with col2:
        fig_stress_disorder = px.box(
            df,
            x='Sleep Disorder',
            y='Stress Level',
            title='Stress Level Distribution by Disorder',
            color='Sleep Disorder',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_stress_disorder.update_layout(**get_plotly_layout(), showlegend=False)
        st.plotly_chart(fig_stress_disorder, use_container_width=True)

    st.markdown("### Sleep Disorders by Occupation")
    occupation_disorder = pd.crosstab(df['Occupation'], df['Sleep Disorder'], normalize='index') * 100
    fig_occupation_disorder = go.Figure()

    for disorder in occupation_disorder.columns:
        fig_occupation_disorder.add_trace(go.Bar(
            name=disorder,
            x=occupation_disorder.index,
            y=occupation_disorder[disorder],
            text=occupation_disorder[disorder].round(1),
            textposition='auto'
        ))
    fig_occupation_disorder.update_layout(
        barmode='stack',
        title='Sleep Disorder Distribution by Occupation (%)',
        xaxis_title='Occupation',
        yaxis_title='Percentage',
        height=500,
        **get_plotly_layout()
    )
    fig_occupation_disorder.update_xaxes(tickangle=45)
    st.plotly_chart(fig_occupation_disorder, use_container_width=True)

    st.markdown("### Statistical Summary by Sleep Disorder")
    st.dataframe(
        disorder_profile.style.background_gradient(cmap='RdYlGn', subset=['Quality of Sleep']),
        use_container_width=True
    )

# -------
# Footer
# -------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <span style='font-size: 2rem;'>🐱</span>
    <p style='color: #A0AEC0; font-size: 0.9rem; margin-top: 0.5rem;'>
        Deep diving into data patterns
    </p>
</div>
""", unsafe_allow_html=True)
