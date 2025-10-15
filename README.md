# Sleep Health and Lifestyle Analysis

## Project Overview

This project analyzes the relationship between lifestyle factors (physical activity, stress, diet, screen time) and sleep quality to identify individuals at risk for sleep disorders. The analysis addresses the neuroscience research question: "How do behavioral patterns affect sleep architecture and cognitive recovery?"

**Research Question:** How do behavioral patterns affect sleep architecture and cognitive recovery?

## Dataset Description

**Source:** Sleep Health and Lifestyle Dataset - Kaggle  
**URL:** https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset  
**Dimensions:** 374 observations × 13 variables

### Variables
- Person ID, Gender, Age, Occupation
- Sleep Duration, Quality of Sleep
- Physical Activity Level, Daily Steps
- Stress Level, BMI Category
- Blood Pressure, Heart Rate
- Sleep Disorder

## Rationale for Dataset Selection

The selection of this dataset was motivated by several key considerations:

1. **Public Health Relevance:** Sleep health represents a critical yet often overlooked dimension of overall wellbeing, with documented effects on cognitive function, emotional regulation, and physical health outcomes.

2. **Multidimensional Analysis Potential:** The dataset incorporates diverse lifestyle factors (activity patterns, stress levels, occupational characteristics) alongside physiological metrics (heart rate, blood pressure), enabling comprehensive analysis of sleep health determinants.

3. **Translational Value:** Unlike purely theoretical datasets, this data enables direct translation to practical recommendations that individuals and healthcare providers can implement to improve sleep quality outcomes.

4. **Neuroscience Connection:** The dataset bridges behavioral patterns with sleep architecture, addressing fundamental questions about how daily habits impact cognitive recovery and neurological health.

5. **Appropriate Complexity:** With 374 observations and 13 features, the dataset provides sufficient complexity for meaningful statistical analysis while remaining manageable for semester-long project scope.

## Initial and Exploratory Data Analysis Findings

### Data Quality Assessment

**Data Integrity:** The dataset exhibits high quality with no missing values in core features, requiring minimal preprocessing interventions. Demographic representation is balanced across age groups (27-59 years) and gender categories. The sample includes representation from 11 distinct occupational categories, ranging from healthcare professionals to engineering roles.

### Key Statistical Findings

#### Sleep Duration Patterns
- Mean sleep duration: 7.13 hours (aligns with clinical recommendations of 7-9 hours)
- Range: 5.8 to 8.5 hours
- Distribution: Majority of participants cluster around 6-8 hours, with minimal extreme cases

#### Sleep Quality Distribution
- Mean quality score: 7.31 out of 10
- Strong positive correlation with sleep duration (r = 0.88)
- Critical finding: Quality demonstrates non-linear relationship with duration, declining significantly below 6 hours and above 9 hours

#### Lifestyle Factor Impact Analysis

**Stress Level Relationships:**
- Strong negative correlation with sleep quality (r = -0.89)
- Participants reporting stress levels above 7 demonstrate 32% lower quality scores
- Statistically significant predictor of sleep outcomes

**Physical Activity Correlations:**
- Moderate positive correlation with sleep quality (r = 0.46)
- Individuals with activity levels exceeding 60 report significantly better sleep outcomes
- Effect size suggests meaningful clinical relevance

**Daily Steps Analysis:**
- Weak positive correlation (r = 0.23)
- Threshold effect observed: exceeding 7,000 steps shows measurable improvement
- May serve as accessible intervention target

#### Sleep Disorder Prevalence

**Distribution:**
- No disorder: 48% of sample
- Sleep Apnea: 38% of sample
- Insomnia: 14% of sample

**Risk Factor Identification:**
- Elevated BMI demonstrates strong association with Sleep Apnea diagnosis
- High stress levels correlate with Insomnia presentation
- Occupational factors show significant variation: nurses and sales representatives demonstrate elevated disorder rates

#### Physiological Metrics

**Cardiovascular Indicators:**
- Mean heart rate: 70 beats per minute (within normal range)
- Blood pressure: Majority of participants exhibit normal ranges (120/80 mmHg)

**Body Mass Index Distribution:**
- Normal weight: 42%
- Overweight: 35%
- Obese: 23%

### Notable Research Findings

Several unexpected patterns emerged from the exploratory analysis:

1. **Age Factor:** Age demonstrates minimal predictive power for sleep quality in this cohort (r = -0.08), contrary to common assumptions about age-related sleep decline.

2. **Gender Differences:** Gender differences in sleep duration and quality scores are statistically insignificant, suggesting lifestyle factors supersede biological sex in this sample.

3. **Occupational Impact:** Occupation emerges as a significant predictor of sleep patterns, with shift work showing particularly poor outcomes. This finding warrants further investigation into work-schedule effects on sleep architecture.

## Data Preprocessing Methodology

### Data Cleaning Procedures

- Verification confirmed absence of missing values in critical columns
- Null values in Sleep Disorder column treated as distinct category ("None")
- Duplicate record check performed (none identified)
- Data type validation completed for all variables

### Feature Engineering

**Blood Pressure Decomposition:** Original 'Blood Pressure' variable separated into 'Systolic_BP' and 'Diastolic_BP' components to enable independent analysis of hypertension indicators.

**Categorical Binning:** 
- Age groups created using clinically relevant boundaries (under 30, 30-40, 40-50, 50+)
- Sleep duration categorized using evidence-based clinical thresholds
- Physical activity levels classified into quartile-based categories (Low, Medium, High, Very High)

### Encoding Strategies

**Label Encoding:** Applied to ordinal and nominal features using appropriate strategies:
- Gender: Binary encoding (0/1)
- BMI Category: Ordinal encoding preserving natural order (Normal < Overweight < Obese)
- Occupation: Numerical codes assigned for algorithmic processing

**Rationale:** This encoding approach preserves inherent ordinality in BMI categories while creating numerical representations suitable for machine learning algorithms.

### Normalization and Scaling

**Standardization:** StandardScaler applied to continuous variables to ensure unit-invariant analysis:
- Demographic: Age
- Sleep metrics: Sleep Duration
- Lifestyle factors: Physical Activity Level, Stress Level, Daily Steps
- Physiological measures: Heart Rate, Systolic BP, Diastolic BP

**Justification:** Standardization ensures all features operate on comparable scales, critical for distance-based algorithms and preventing feature dominance based solely on measurement units.

### Categorical Variable Treatment

- Sleep Disorder retained as target variable with three levels (None, Sleep Apnea, Insomnia)
- One-hot encoding implemented for multi-class occupation analysis
- Original columns preserved for interpretability in visualization outputs

### Data Validation

**Quality Control Measures:**
- Outlier detection using Interquartile Range (IQR) method
- Physiological range validation (Heart Rate: 60-90 bpm, Blood Pressure: clinically normal ranges)
- Logical consistency verification (e.g., sleep duration constrained to 4-10 hours)

## Streamlit Application Development

### Application Architecture

The interactive web application implements a multi-page architecture with the following components:

#### Home Page
- Project overview and research objectives
- Summary statistics dashboard displaying key metrics
- Demographic distribution visualizations
- Sleep disorder prevalence breakdown
- Descriptive statistics table

#### Data Explorer
- Dynamic filtering interface with age range, gender, and occupation selectors
- Real-time data table updates based on filter criteria
- CSV export functionality for filtered datasets
- Distribution visualizations responsive to filter selections

#### Deep Dive Analysis
- Correlation matrix heatmap displaying all inter-variable relationships
- Demographic trend analysis including sleep duration by occupation and quality by age group
- Interactive scatter plot interface enabling user-defined axis selection
- Color-coding options by categorical variables
- Hover functionality displaying detailed observation information

#### Sleep Quality Predictor
- Comprehensive input form collecting:
  - Demographic information (age, gender, occupation, BMI)
  - Sleep patterns (duration)
  - Lifestyle factors (activity level, daily steps, stress level)
  - Physiological metrics (heart rate, blood pressure)
- Rule-based prediction engine generating quality scores
- Results display including predicted quality score, risk level assessment, and comparison to population average
- Personalized recommendation system based on input parameters

#### Insights Dashboard
- Curated presentation of four key research findings
- Supporting visualizations for each identified pattern
- Comprehensive summary statistics table

### Technical Implementation

**Performance Optimization:**
- Implementation of Streamlit's caching decorator (@st.cache_data) for data loading and preprocessing operations
- Efficient pandas vectorized operations throughout analysis pipeline
- Lazy loading strategies for computationally intensive visualizations

**Visualization Framework:**
- Plotly library utilized for interactive visualizations with zoom, pan, and hover capabilities
- Seaborn and Matplotlib employed for static publication-quality figures
- Custom CSS styling for professional interface appearance

**User Experience Design:**
- Responsive column layouts adapting to various screen dimensions
- Tab-based organization for content hierarchy
- Metric cards for key performance indicators
- Sidebar navigation for intuitive page transitions
- Loading indicators for asynchronous operations

## Project Structure

```
sleep-health-analysis/
│
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── (Planned additions)
    ├── notebooks/
    │   └── eda.ipynb              # Detailed exploratory analysis
    ├── models/
    │   └── sleep_predictor.pkl    # Trained ML model
    └── utils/
        └── preprocessing.py        # Preprocessing functions
```

## Technology Stack

- **Python 3.9+:** Primary programming language
- **Streamlit:** Web application framework
- **Pandas:** Data manipulation and analysis
- **Plotly:** Interactive data visualization
- **Scikit-learn:** Preprocessing and machine learning functionality
- **NumPy:** Numerical computation
- **Seaborn/Matplotlib:** Statistical visualization

## Installation and Execution

### Local Deployment

```bash
# Clone repository
git clone https://github.com/Yashwitha-7/sleep-health-analysis.git
cd sleep-health-analysis

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`

## Deployment

**Production URL:** [To be added upon deployment]

**Deployment Platform:** Streamlit Cloud  
**Deployment Process:**
1. Repository connection via GitHub integration
2. Main file specification: app.py
3. Automatic dependency installation from requirements.txt
4. Continuous deployment on repository updates

## Future Development Roadmap

### Planned Enhancements for Final Submission
- Implementation of supervised learning models (Random Forest, Gradient Boosting)
- Cross-validation and performance metric reporting
- Prediction confidence intervals
- Advanced visualization techniques (3D projections, network graphs)
- User data upload and analysis functionality
- Comparative analysis tool for similar demographic profiles
- PDF report generation capability
- Educational content modules on sleep science

### Extended Features (Post-Course)
- Temporal analysis capabilities for longitudinal data
- Unsupervised clustering for sleep profile identification
- Feature importance analysis and visualization
- Experimental design framework for lifestyle interventions
- Wearable device data integration

## Academic Learnings

### Technical Competencies Developed
1. End-to-end data analysis pipeline construction
2. Interactive web application development using Streamlit framework
3. Advanced data visualization and storytelling techniques
4. Data preprocessing methodologies including encoding, scaling, and feature engineering
5. User-centered design principles for non-technical audiences

### Domain Knowledge Acquired
1. Sleep science fundamentals and factors affecting sleep architecture
2. Health data analysis methodologies and physiological metric interpretation
3. Behavioral pattern analysis and health outcome relationships
4. Risk assessment and predictive modeling in health contexts

## References

**Dataset Source:**  
Sleep Health and Lifestyle Dataset. Kaggle.  
Available at: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

**Additional Resources:**
- National Sleep Foundation Clinical Guidelines
- Centers for Disease Control and Prevention: Sleep and Sleep Disorders
- Streamlit Documentation and Best Practices

## Author Information

**Name:** Yashwitha  
**Institution:** Michigan State University  
**Course:** CMSE830 - Foundations of Data Science  
**Semester:** Fall 2025  
**GitHub:** [@Yashwitha-7](https://github.com/Yashwitha-7)

## Acknowledgments

This project was developed with assistance from Claude (Anthropic), an AI assistant that provided guidance on:
- Project structure and organization
- Streamlit application architecture and implementation
- Data visualization best practices
- Documentation formatting and academic writing conventions
- Code optimization and debugging support

Claude served as a technical consultation resource similar to office hours or tutoring support. All conceptual framework decisions, data analysis interpretations, research insights, and academic content were developed independently by the author.

## License

This project is submitted as coursework for academic evaluation. All rights reserved. The dataset is used under Kaggle's terms of service for educational purposes.

---

**Project Status:** In Progress - Midterm Submission  
**Last Updated:** October 2025

**Note:** This project represents work completed through the midterm evaluation period. Additional features, analysis, and machine learning model implementation are planned for final submission.
