import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Page + Theme Configuration
# ---------------------------
st.set_page_config(page_title="Sleep Quality Predictor", page_icon="🎯", layout="wide")

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

# -----------------
# Train the model
# -----------------
@st.cache_resource
def train_model():
    df_model = df.copy()

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()

    df_model['Gender_Encoded'] = le_gender.fit_transform(df_model['Gender'])
    df_model['Occupation_Encoded'] = le_occupation.fit_transform(df_model['Occupation'])
    df_model['BMI_Encoded'] = le_bmi.fit_transform(df_model['BMI Category'])

    # Features for prediction
    features = ['Age', 'Gender_Encoded', 'Occupation_Encoded', 'Sleep Duration',
                'Physical Activity Level', 'Stress Level', 'BMI_Encoded',
                'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']

    X = df_model[features]
    y = df_model['Quality of Sleep']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_gender, le_occupation, le_bmi, features

model, le_gender, le_occupation, le_bmi, feature_names = train_model()

# -----------
# Header
# -----------
st.title("Sleep Quality Predictor")
st.markdown("### Personalized Sleep Quality Assessment Tool")

# Gradient intro card (white text enforced)
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
    <h3 style='color: white !important; margin: 0; padding-bottom: 0.5rem;'>How Does This Work?</h3>
    <p style='margin: 0; font-size: 1rem; color: white !important;'>
        This tool uses machine learning trained on 374 individuals to predict your sleep quality score.
        Enter your lifestyle and health metrics below to receive a personalized assessment and recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------
# Inputs
# -----------
st.markdown("## Enter Your Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Demographics")
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    gender = st.selectbox("Gender", df['Gender'].unique())
    occupation = st.selectbox("Occupation", sorted(df['Occupation'].unique()))

with col2:
    st.markdown("### Sleep & Lifestyle")
    sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 12.0, 7.0, 0.5)
    physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 60, 5)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    daily_steps = st.number_input("Daily Steps", 0, 20000, 7000, 500)

with col3:
    st.markdown("### Health Metrics")
    bmi_category = st.selectbox("BMI Category", df['BMI Category'].unique())
    heart_rate = st.number_input("Resting Heart Rate (BPM)", 40, 120, 70)
    systolic_bp = st.number_input("Systolic Blood Pressure", 90, 180, 120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", 60, 120, 80)

st.markdown("---")

# -----------
# Prediction
# -----------
if st.button("Calculate My Sleep Quality Score", type="primary"):
    # Encode inputs (assumes chosen values exist in fitted encoders)
    gender_encoded = le_gender.transform([gender])[0]
    occupation_encoded = le_occupation.transform([occupation])[0]
    bmi_encoded = le_bmi.transform([bmi_category])[0]

    input_data = np.array([[age, gender_encoded, occupation_encoded, sleep_duration,
                           physical_activity, stress_level, bmi_encoded,
                           heart_rate, daily_steps, systolic_bp, diastolic_bp]])

    predicted_quality = model.predict(input_data)[0]
    predicted_quality = float(np.clip(predicted_quality, 1, 10))  # constrain to 1..10

    # Risk level buckets
    if predicted_quality >= 8:
        risk_level = "Low Risk"
        risk_color = "#2ECC71"
        risk_message = "Excellent sleep quality! Keep up your healthy habits."
    elif predicted_quality >= 6:
        risk_level = "Moderate Risk"
        risk_color = "#F39C12"
        risk_message = "Good sleep quality, but there's room for improvement."
    else:
        risk_level = "High Risk"
        risk_color = "#E74C3C"
        risk_message = "Your sleep quality needs attention. Consider the recommendations below."

    # -----------------
    # Results cards
    # -----------------
    st.markdown("## Your Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        # Dark card with colored border
        st.markdown(f"""
        <div style='background-color: #1A1D24; padding: 2rem; border-radius: 10px;
                    text-align: center; border: 3px solid {risk_color}; color: #E8E8E8;'>
            <h2 style='color: {risk_color}; margin: 0; font-size: 3rem;'>{predicted_quality:.1f}</h2>
            <p style='margin: 0.5rem 0 0 0; color: #A0AEC0; font-size: 1.2rem;'>Sleep Quality Score</p>
            <p style='margin: 0.5rem 0 0 0; color: #A0AEC0; font-size: 0.9rem;'>Out of 10</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Subtle gradient with white text
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {risk_color}33, {risk_color}55);
                    padding: 2rem; border-radius: 10px; text-align: center; color: white;'>
            <h3 style='color: white !important; margin: 0;'>{risk_level}</h3>
            <p style='margin: 1rem 0 0 0; color: white !important;'>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Comparison vs dataset average on dark card
        avg_quality = df['Quality of Sleep'].mean()
        difference = predicted_quality - avg_quality
        comparison_text = "above" if difference > 0 else "below"
        comparison_color = "#2ECC71" if difference > 0 else "#E74C3C"

        st.markdown(f"""
        <div style='background-color: #1A1D24; padding: 2rem; border-radius: 10px; text-align: center; color: #E8E8E8;'>
            <h4 style='margin: 0; color: #E8E8E8;'>vs Dataset Average</h4>
            <h2 style='color: {comparison_color}; margin: 0.5rem 0;'>{abs(difference):.1f}</h2>
            <p style='margin: 0; color: #A0AEC0;'>points {comparison_text} average</p>
            <p style='margin: 0.5rem 0 0 0; color: #A0AEC0; font-size: 0.9rem;'>
                Dataset avg: {avg_quality:.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------
    # Gauge chart
    # -----------------
    st.markdown("## Visual Score Representation")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_quality,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Quality Score", 'font': {'size': 24}},
        delta={'reference': avg_quality, 'increasing': {'color': "#2ECC71"}, 'decreasing': {'color': "#E74C3C"}},
        gauge={
            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "#A0AEC0"},
            'bar': {'color': risk_color},
            'bgcolor': "#1A1D24",
            'borderwidth': 2,
            'bordercolor': "#2D3748",
            'steps': [
                {'range': [0, 4], 'color': '#3A1E1E'},     # dark red zone
                {'range': [4, 7], 'color': '#3A321A'},     # dark amber zone
                {'range': [7, 10], 'color': '#1A3A30'}     # dark green zone
            ],
            'threshold': {
                'line': {'color': "#E8E8E8", 'width': 4},
                'thickness': 0.75,
                'value': avg_quality
            }
        }
    ))
    fig_gauge.update_layout(height=400, **get_plotly_layout())
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # -----------------
    # Personalized recommendations
    # -----------------
    st.markdown("## Personalized Recommendations")

    recommendations = []

    # Sleep duration recs
    if sleep_duration < 7:
        recommendations.append({
            'category': 'Sleep Duration',
            'icon': '🛏️',
            'message': f'You are sleeping {sleep_duration} hours per night. Aim for 7-9 hours for optimal health.',
            'action': 'Set a consistent bedtime and wake time. Avoid screens 1 hour before bed.'
        })
    elif sleep_duration > 9:
        recommendations.append({
            'category': 'Sleep Duration',
            'icon': '⏰',
            'message': f'You are sleeping {sleep_duration} hours per night, which is above the recommended range.',
            'action': 'Consider if you might have an underlying sleep disorder. Consult a healthcare provider.'
        })

    # Physical activity
    if physical_activity < 30:
        recommendations.append({
            'category': 'Physical Activity',
            'icon': '🏃',
            'message': 'Your physical activity level is low. Regular exercise improves sleep quality.',
            'action': 'Start with 30 minutes of moderate exercise daily. Even a brisk walk helps!'
        })

    # Stress
    if stress_level >= 7:
        recommendations.append({
            'category': 'Stress Management',
            'icon': '🧘',
            'message': 'Your stress level is high, which significantly impacts sleep quality.',
            'action': 'Practice relaxation techniques: meditation, deep breathing, or yoga before bed.'
        })

    # Steps
    if daily_steps < 5000:
        recommendations.append({
            'category': 'Daily Activity',
            'icon': '👟',
            'message': f'You are averaging {daily_steps} steps per day. More movement improves sleep.',
            'action': 'Gradually increase to 7,000-10,000 steps daily. Take short walks throughout the day.'
        })

    # Heart rate
    if heart_rate > 80:
        recommendations.append({
            'category': 'Heart Health',
            'icon': '❤️',
            'message': 'Your resting heart rate is elevated. This can affect sleep quality.',
            'action': 'Regular cardiovascular exercise can lower resting heart rate. Consult your doctor.'
        })

    # Blood pressure
    if systolic_bp > 130 or diastolic_bp > 85:
        recommendations.append({
            'category': 'Blood Pressure',
            'icon': '🩺',
            'message': 'Your blood pressure is elevated. This can impact sleep and overall health.',
            'action': 'Reduce sodium intake, exercise regularly, and monitor your BP. See your healthcare provider.'
        })

    # BMI
    if 'Obese' in bmi_category or 'Overweight' in bmi_category:
        recommendations.append({
            'category': 'Weight Management',
            'icon': '⚖️',
            'message': 'Your BMI category may be affecting your sleep quality.',
            'action': 'Focus on balanced nutrition and regular physical activity. Consult a nutritionist.'
        })

    # Display recommendations on dark cards
    if recommendations:
        for rec in recommendations:
            st.markdown(f"""
            <div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px;
                        margin-bottom: 1rem; border-left: 4px solid #3498DB; color: #E8E8E8;'>
                <h4 style='margin: 0; color: #E8E8E8;'>
                    <span style='font-size: 1.5rem;'>{rec['icon']}</span> {rec['category']}
                </h4>
                <p style='margin: 0.5rem 0; color: #A0AEC0;'><strong>Insight:</strong> {rec['message']}</p>
                <p style='margin: 0.5rem 0 0 0; color: #E8E8E8;'><strong>Action:</strong> {rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 2rem; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='color: white !important; margin: 0;'>🎉 Excellent Lifestyle Habits!</h3>
            <p style='margin: 1rem 0 0 0; font-size: 1.1rem; color: white !important;'>
                Your metrics are within healthy ranges. Keep up the great work maintaining these habits!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------
    # Your metrics vs dataset averages
    # -----------------
    st.markdown("## How Your Inputs Compare")

    comparison_data = pd.DataFrame({
        'Metric': ['Sleep Duration', 'Physical Activity', 'Stress Level',
                   'Daily Steps', 'Heart Rate'],
        'Your Value': [sleep_duration, physical_activity, stress_level,
                       daily_steps, heart_rate],
        'Dataset Average': [
            df['Sleep Duration'].mean(),
            df['Physical Activity Level'].mean(),
            df['Stress Level'].mean(),
            df['Daily Steps'].mean(),
            df['Heart Rate'].mean()
        ]
    })

    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='Your Value',
        x=comparison_data['Metric'],
        y=comparison_data['Your Value'],
        marker_color='#3498DB'
    ))
    fig_comparison.add_trace(go.Bar(
        name='Dataset Average',
        x=comparison_data['Metric'],
        y=comparison_data['Dataset Average'],
        marker_color='#95A5A6'
    ))
    fig_comparison.update_layout(
        barmode='group',
        title='Your Metrics vs Dataset Averages',
        xaxis_title='',
        yaxis_title='Value',
        height=400,
        **get_plotly_layout()
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

else:
    # Placeholder when no prediction yet (dark card)
    st.markdown("""
    <div style='background-color: #1A1D24; padding: 3rem; border-radius: 10px; text-align: center; color: #E8E8E8;'>
        <span style='font-size: 4rem;'>🐱</span>
        <h3 style='color: #E8E8E8; margin: 1rem 0;'>Ready When You Are!</h3>
        <p style='color: #A0AEC0;'>Fill in your information above and click the button to get your personalized sleep quality assessment.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------
# Footer
# -----------
st.markdown("---")
st.markdown("""
<div style='background-color: #1A1D24; padding: 1.5rem; border-radius: 10px; margin-top: 2rem; border-left: 4px solid #3498DB; color: #E8E8E8;'>
    <p style='margin: 0; text-align: center;'>
        <strong>Disclaimer:</strong> This tool provides estimates based on statistical models and should not replace
        professional medical advice. Consult healthcare providers for sleep concerns.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <span style='font-size: 2rem;'>🐱</span>
    <p style='color: #A0AEC0; font-size: 0.9rem; margin-top: 0.5rem;'>
        Predicting your sleep health
    </p>
</div>
""", unsafe_allow_html=True)
