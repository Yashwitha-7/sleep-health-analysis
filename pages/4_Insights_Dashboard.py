import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ---------------------------
# Page + Theme Configuration
# ---------------------------
st.set_page_config(page_title="Insights Dashboard", page_icon="💡", layout="wide")

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

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/sleep_health_cleaned.csv')
    with open('data/summary_statistics.json', 'r') as f:
        summary_stats = json.load(f)
    return df, summary_stats

df, summary_stats = load_data()

# Extract BP components if not already present
if 'Systolic_BP' not in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# ---------------------------
# Header
# ---------------------------
st.title("Insights Dashboard")
st.markdown("### Key Findings and Actionable Recommendations")

# Gradient hero (white text enforced)
st.markdown("""
<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
    <h3 style='color: white !important; margin: 0; padding-bottom: 0.5rem;'>Data-Driven Sleep Health Insights</h3>
    <p style='margin: 0; font-size: 1rem; color: white !important;'>
        This dashboard synthesizes key findings from our comprehensive analysis of 374 individuals.
        Discover evidence-based patterns to improve your sleep quality.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Executive Summary
# ---------------------------
st.markdown("## Executive Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                border-left: 4px solid #3498DB; height: 180px; color: #E8E8E8;'>
        <h4 style='color: #3498DB; margin: 0; font-size: 1rem;'>Sleep Duration</h4>
        <h2 style='color: #E8E8E8; margin: 0.5rem 0;'>7.1 hrs</h2>
        <p style='color: #A0AEC0; margin: 0; font-size: 0.9rem;'>
            Average sleep duration is slightly below the recommended 7-9 hours
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    disorder_rate = (len(df[df['Sleep Disorder'] != 'None']) / len(df)) * 100
    st.markdown(f"""
    <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                border-left: 4px solid #E74C3C; height: 180px; color: #E8E8E8;'>
        <h4 style='color: #E74C3C; margin: 0; font-size: 1rem;'>Sleep Disorders</h4>
        <h2 style='color: #E8E8E8; margin: 0.5rem 0;'>{disorder_rate:.1f}%</h2>
        <p style='color: #A0AEC0; margin: 0; font-size: 0.9rem;'>
            Of participants have diagnosed sleep disorders requiring attention
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    correlation = df[['Physical Activity Level', 'Quality of Sleep']].corr().iloc[0, 1]
    st.markdown(f"""
    <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                border-left: 4px solid #2ECC71; height: 180px; color: #E8E8E8;'>
        <h4 style='color: #2ECC71; margin: 0; font-size: 1rem;'>Activity Impact</h4>
        <h2 style='color: #E8E8E8; margin: 0.5rem 0;'>+{correlation:.2f}</h2>
        <p style='color: #A0AEC0; margin: 0; font-size: 0.9rem;'>
            Strong positive correlation between physical activity and sleep quality
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    stress_correlation = df[['Stress Level', 'Quality of Sleep']].corr().iloc[0, 1]
    st.markdown(f"""
    <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                border-left: 4px solid #F39C12; height: 180px; color: #E8E8E8;'>
        <h4 style='color: #F39C12; margin: 0; font-size: 1rem;'>Stress Impact</h4>
        <h2 style='color: #E8E8E8; margin: 0.5rem 0;'>{stress_correlation:.2f}</h2>
        <p style='color: #A0AEC0; margin: 0; font-size: 0.9rem;'>
            Significant negative correlation between stress and sleep quality
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Top 5 Findings (dark cards)
# ---------------------------
st.markdown("## Top 5 Key Findings")
findings = [
    {'number': '1', 'title': 'Physical Activity is the Strongest Predictor',
     'description': 'Higher activity levels (>60 min/day) associate with much better sleep quality.',
     'color': '#3498DB', 'icon': '🏃'},
    {'number': '2', 'title': 'Stress Management is Critical',
     'description': 'High stress (7+ on scale) lowers sleep quality by ~2+ points on average.',
     'color': '#E74C3C', 'icon': '😰'},
    {'number': '3', 'title': 'Occupation Matters',
     'description': 'Nurses and salespeople show higher stress and lower sleep quality; engineers and teachers fare better.',
     'color': '#9B59B6', 'icon': '💼'},
    {'number': '4', 'title': 'Sleep Duration Sweet Spot',
     'description': 'Optimal sleep quality occurs between 7–8 hours; <6 or >9 correlates with lower scores.',
     'color': '#2ECC71', 'icon': '⏰'},
    {'number': '5', 'title': 'BMI and Sleep Disorders Link',
     'description': 'Obesity is linked with higher prevalence of sleep apnea.',
     'color': '#F39C12', 'icon': '⚖️'}
]
for f in findings:
    st.markdown(f"""
    <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                margin-bottom: 1rem; border-left: 5px solid {f['color']}; color: #E8E8E8;'>
        <div style='display: flex; align-items: start;'>
            <div style='background-color: {f['color']}; color: white;
                        width: 40px; height: 40px; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center;
                        font-weight: bold; font-size: 1.2rem; margin-right: 1rem; flex-shrink: 0;'>
                {f['number']}
            </div>
            <div style='flex-grow: 1;'>
                <h4 style='margin: 0; color: #E8E8E8;'>
                    <span style='font-size: 1.3rem; margin-right: 0.5rem;'>{f['icon']}</span>
                    {f['title']}
                </h4>
                <p style='margin: 0.5rem 0 0 0; color: #A0AEC0; line-height: 1.6;'>
                    {f['description']}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Visual Insights
# ---------------------------
st.markdown("## Visual Insights")
tab1, tab2, tab3 = st.tabs(["Lifestyle Factors", "Demographics & Health", "Sleep Disorders"])

with tab1:
    st.markdown("### Impact of Lifestyle Factors on Sleep Quality")
    col1, col2 = st.columns(2)

    with col1:
        fig_activity = px.scatter(
            df,
            x='Physical Activity Level',
            y='Quality of Sleep',
            color='Stress Level',
            size='Daily Steps',
            title='Physical Activity Impact on Sleep Quality',
            labels={'Physical Activity Level': 'Physical Activity (min/day)', 'Quality of Sleep': 'Sleep Quality Score'},
            color_continuous_scale='RdYlGn_r',
            trendline='ols'
        )
        fig_activity.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_activity, use_container_width=True)

        st.markdown("""
        **Key Insight:** More activity correlates with better sleep quality. Even +15–30 minutes daily helps.
        """)

    with col2:
        stress_bins = pd.cut(df['Stress Level'], bins=[0, 3, 5, 7, 10],
                             labels=['Low (1-3)', 'Moderate (4-5)', 'High (6-7)', 'Very High (8-10)'])
        stress_quality = df.groupby(stress_bins)['Quality of Sleep'].mean().reset_index()

        fig_stress = px.bar(
            stress_quality,
            x='Stress Level',
            y='Quality of Sleep',
            title='Sleep Quality by Stress Level Category',
            color='Quality of Sleep',
            color_continuous_scale='RdYlGn',
            text='Quality of Sleep'
        )
        fig_stress.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_stress.update_layout(**get_plotly_layout(), showlegend=False)
        st.plotly_chart(fig_stress, use_container_width=True)

        st.markdown("""
        **Key Insight:** Sleep quality decreases steadily as stress rises—stress management is essential.
        """)

    # Daily steps analysis
    st.markdown("### Daily Steps and Sleep Quality Relationship")

    df['Steps_Category'] = pd.cut(
        df['Daily Steps'],
        bins=[0, 5000, 7500, 10000, 15000],
        labels=['Sedentary\n(<5k)', 'Low Active\n(5k-7.5k)', 'Active\n(7.5k-10k)', 'Very Active\n(>10k)']
    )

    steps_analysis = df.groupby('Steps_Category').agg({
        'Quality of Sleep': 'mean',
        'Sleep Duration': 'mean',
        'Person ID': 'count'
    }).reset_index()
    steps_analysis.columns = ['Steps_Category', 'Avg_Sleep_Quality', 'Avg_Sleep_Duration', 'Count']

    fig_steps = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sleep Quality by Activity Level', 'Sleep Duration by Activity Level'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    fig_steps.add_trace(
        go.Bar(x=steps_analysis['Steps_Category'],
               y=steps_analysis['Avg_Sleep_Quality'],
               marker_color='#3498DB',
               text=steps_analysis['Avg_Sleep_Quality'].round(1),
               textposition='outside',
               name='Sleep Quality'),
        row=1, col=1
    )
    fig_steps.add_trace(
        go.Bar(x=steps_analysis['Steps_Category'],
               y=steps_analysis['Avg_Sleep_Duration'],
               marker_color='#2ECC71',
               text=steps_analysis['Avg_Sleep_Duration'].round(1),
               textposition='outside',
               name='Sleep Duration'),
        row=1, col=2
    )
    fig_steps.update_layout(height=400, showlegend=False, **get_plotly_layout())
    st.plotly_chart(fig_steps, use_container_width=True)

with tab2:
    st.markdown("### Demographic and Health Patterns")
    col1, col2 = st.columns(2)

    with col1:
        df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70],
                                 labels=['20-30', '30-40', '40-50', '50-60', '60+'])

        age_metrics = df.groupby('Age_Group').agg({
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean'
        }).reset_index()

        fig_age_radar = go.Figure()
        for age_group in age_metrics['Age_Group']:
            values = age_metrics[age_metrics['Age_Group'] == age_group][
                ['Quality of Sleep', 'Physical Activity Level']].values[0]
            stress_inverted = 10 - age_metrics[age_metrics['Age_Group'] == age_group]['Stress Level'].values[0]
            values = list(values) + [stress_inverted]
            fig_age_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=['Sleep Quality', 'Physical Activity', 'Low Stress', 'Sleep Quality'],
                name=str(age_group),
                fill='toself'
            ))

        fig_age_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#2D3748'),
                       angularaxis=dict(gridcolor='#2D3748')),
            title='Health Metrics Across Age Groups',
            showlegend=True,
            height=450,
            **get_plotly_layout()
        )
        st.plotly_chart(fig_age_radar, use_container_width=True)

    with col2:
        bmi_disorder = pd.crosstab(df['BMI Category'], df['Sleep Disorder'], normalize='index') * 100
        fig_bmi = go.Figure()
        for disorder in bmi_disorder.columns:
            fig_bmi.add_trace(go.Bar(
                name=disorder,
                x=bmi_disorder.index,
                y=bmi_disorder[disorder],
                text=bmi_disorder[disorder].round(1),
                textposition='auto'
            ))
        fig_bmi.update_layout(
            barmode='stack',
            title='Sleep Disorder Prevalence by BMI Category',
            xaxis_title='BMI Category',
            yaxis_title='Percentage (%)',
            height=450,
            **get_plotly_layout()
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    # Gender comparison (radar)
    st.markdown("### Gender-Based Sleep Health Comparison")
    gender_comparison = df.groupby('Gender').agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Stress Level': 'mean',
        'Physical Activity Level': 'mean',
        'Heart Rate': 'mean',
        'Daily Steps': 'mean'
    }).round(2)

    fig_gender_compare = go.Figure()
    metrics = ['Sleep Duration', 'Quality of Sleep', 'Stress Level',
               'Physical Activity Level', 'Heart Rate', 'Daily Steps']

    for gender in gender_comparison.index:
        normalized_values = []
        for metric in metrics:
            if metric == 'Daily Steps':
                normalized_values.append(gender_comparison.loc[gender, metric] / 1000)
            elif metric == 'Heart Rate':
                normalized_values.append(gender_comparison.loc[gender, metric] / 10)
            else:
                normalized_values.append(gender_comparison.loc[gender, metric])

        fig_gender_compare.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],
            theta=['Sleep Duration', 'Sleep Quality', 'Stress Level',
                   'Activity Level', 'Heart Rate/10', 'Steps/1000', 'Sleep Duration'],
            name=str(gender),
            fill='toself'
        ))

    fig_gender_compare.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#2D3748'),
                   angularaxis=dict(gridcolor='#2D3748')),
        title='Comprehensive Gender Comparison (Normalized)',
        showlegend=True,
        height=500,
        **get_plotly_layout()
    )
    st.plotly_chart(fig_gender_compare, use_container_width=True)

with tab3:
    st.markdown("### Sleep Disorder Deep Dive")
    col1, col2 = st.columns([1, 2])

    with col1:
        disorder_counts = df['Sleep Disorder'].value_counts()
        fig_disorder_donut = go.Figure(data=[go.Pie(
            labels=disorder_counts.index,
            values=disorder_counts.values,
            hole=.4,
            marker=dict(colors=['#2ECC71', '#E74C3C', '#F39C12'])
        )])
        fig_disorder_donut.update_layout(
            title='Sleep Disorder Distribution',
            annotations=[dict(text=f"{len(df)}<br>Total", x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=400,
            **get_plotly_layout()
        )
        st.plotly_chart(fig_disorder_donut, use_container_width=True)

    with col2:
        st.markdown("#### Disorder Characteristics")
        disorder_stats = df.groupby('Sleep Disorder').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean',
            'Heart Rate': 'mean',
            'BMI Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        }).round(2)
        st.dataframe(
            disorder_stats.style.background_gradient(cmap='RdYlGn', subset=['Quality of Sleep']),
            use_container_width=True
        )

    st.markdown("### Risk Factor Analysis by Sleep Disorder")
    fig_disorder_factors = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sleep Duration', 'Stress Level', 'Physical Activity', 'Heart Rate'),
        vertical_spacing=0.12
    )
    disorders = df['Sleep Disorder'].unique()
    colors = {'None': '#2ECC71', 'Sleep Apnea': '#E74C3C', 'Insomnia': '#F39C12'}

    for disorder in disorders:
        subset = df[df['Sleep Disorder'] == disorder]
        fig_disorder_factors.add_trace(go.Box(y=subset['Sleep Duration'], name=disorder,
                                              marker_color=colors.get(disorder, '#3498DB'), showlegend=False),
                                       row=1, col=1)
        fig_disorder_factors.add_trace(go.Box(y=subset['Stress Level'], name=disorder,
                                              marker_color=colors.get(disorder, '#3498DB'), showlegend=False),
                                       row=1, col=2)
        fig_disorder_factors.add_trace(go.Box(y=subset['Physical Activity Level'], name=disorder,
                                              marker_color=colors.get(disorder, '#3498DB'), showlegend=False),
                                       row=2, col=1)
        fig_disorder_factors.add_trace(go.Box(y=subset['Heart Rate'], name=disorder,
                                              marker_color=colors.get(disorder, '#3498DB'), showlegend=False),
                                       row=2, col=2)
    fig_disorder_factors.update_layout(height=600, **get_plotly_layout())
    st.plotly_chart(fig_disorder_factors, use_container_width=True)

st.markdown("---")

# ---------------------------
# Interactive Summary: Your Risk Profile
# ---------------------------
st.markdown("## Interactive Summary: Your Risk Profile")
st.markdown("""
<div style='background-color: #1A1D24; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: #E8E8E8;'>
    <p style='margin: 0; color: #A0AEC0;'>
        Adjust the sliders below to see how different lifestyle factors impact predicted sleep quality:
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    sim_activity = st.slider("Physical Activity (min)", 0, 120, 60, key='sim_activity')
    sim_stress = st.slider("Stress Level", 1, 10, 5, key='sim_stress')
with col2:
    sim_sleep_dur = st.slider("Sleep Duration (hrs)", 4.0, 12.0, 7.0, 0.5, key='sim_sleep')
    sim_steps = st.slider("Daily Steps (thousands)", 0, 20, 8, key='sim_steps')
with col3:
    sim_hr = st.slider("Heart Rate (BPM)", 50, 100, 70, key='sim_hr')
    sim_age = st.slider("Age", 20, 70, 35, key='sim_age')

# Heuristic scoring for demo
base_score = 5.0
activity_impact = (sim_activity / 60) * 1.5
stress_impact = -(sim_stress / 10) * 2.0
sleep_impact = 1.0 if 7 <= sim_sleep_dur <= 8 else -0.5
steps_impact = (sim_steps / 10) * 0.8
hr_impact = -0.5 if sim_hr > 80 else 0.3
age_impact = -0.3 if sim_age > 50 else 0.2

predicted_score = base_score + activity_impact + stress_impact + sleep_impact + steps_impact + hr_impact + age_impact
predicted_score = float(max(1, min(10, predicted_score)))

# Gauge visualization (dark)
fig_gauge_summary = go.Figure(go.Indicator(
    mode="gauge+number",
    value=predicted_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Predicted Sleep Quality", 'font': {'size': 24}},
    gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "#A0AEC0"},
        'bar': {'color': "#3498DB"},
        'bgcolor': "#1A1D24",
        'borderwidth': 2,
        'bordercolor': "#2D3748",
        'steps': [
            {'range': [0, 4], 'color': '#3A1E1E'},
            {'range': [4, 7], 'color': '#3A321A'},
            {'range': [7, 10], 'color': '#1A3A30'}
        ],
        'threshold': {'line': {'color': "#E74C3C", 'width': 4}, 'thickness': 0.75, 'value': 8}
    }
))
fig_gauge_summary.update_layout(height=350, **get_plotly_layout())
st.plotly_chart(fig_gauge_summary, use_container_width=True)

# Impact breakdown (dark)
st.markdown("### Factor Contribution Analysis")
contributions = pd.DataFrame({
    'Factor': ['Physical Activity', 'Stress Level', 'Sleep Duration', 'Daily Steps', 'Heart Rate', 'Age'],
    'Impact': [activity_impact, stress_impact, sleep_impact, steps_impact, hr_impact, age_impact]
}).sort_values('Impact', ascending=True)

fig_contributions = px.bar(
    contributions, y='Factor', x='Impact', orientation='h',
    title='How Each Factor Affects Your Sleep Quality Score',
    color='Impact', color_continuous_scale='RdYlGn', range_color=[-3, 3]
)
fig_contributions.update_layout(showlegend=False, **get_plotly_layout(),
                                xaxis_title='Impact on Sleep Quality Score', yaxis_title='')
st.plotly_chart(fig_contributions, use_container_width=True)

st.markdown("---")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <div style='font-size: 4rem; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;'>
        🐱
    </div>
    <h4 style='color: #E8E8E8; margin: 0;'>Sweet dreams and healthy sleep!</h4>
    <p style='color: #A0AEC0; font-size: 0.9rem; margin-top: 0.5rem;'>
        Thank you for exploring our sleep health insights dashboard
    </p>
</div>

<style>
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}
</style>
""", unsafe_allow_html=True)
