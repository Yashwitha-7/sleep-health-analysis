# Sleep Health and Lifestyle Analysis

## Project Overview

This project analyzes the relationship between lifestyle factors (physical activity, stress, diet, screen time) and sleep quality to identify individuals at risk for sleep disorders.

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
- Mean sleep duration: 7.13 hours (slightly below the recommended 7-9 hours)
- Standard deviation: 0.80 hours
- Range: 5.8 to 8.5 hours
- Distribution: Majority of participants cluster around 6-8 hours, with minimal extreme cases
- Sleep deficit (from 8-hour baseline): 0.87 hours on average

#### Sleep Quality Distribution
- Mean quality score: 7.31 out of 10
- Standard deviation: 1.20
- Range: 4-9 on a 10-point scale
- Strong positive correlation with sleep duration (r = 0.88)
- Critical finding: Quality demonstrates non-linear relationship with duration, declining significantly below 6 hours and above 9 hours

#### Lifestyle Factor Impact Analysis

**Stress Level Relationships:**
- Strong negative correlation with sleep quality (r = -0.90)
- Mean stress level: 5.39 out of 10
- Participants reporting stress levels above 7 demonstrate substantially lower quality scores
- Stress emerges as the single strongest predictor of poor sleep quality
- Statistically significant predictor of sleep outcomes across all demographic groups

**Physical Activity Correlations:**
- Moderate positive correlation with sleep quality (r = 0.19)
- Mean activity level: 59.17 minutes per day
- Individuals with activity levels exceeding 60 minutes report significantly better sleep outcomes
- Activity-stress ratio shows importance of balancing exercise with stress management
- Effect size suggests meaningful clinical relevance

**Daily Steps Analysis:**
- Weak positive correlation with sleep quality (r = 0.02)
- Mean daily steps: 6,817
- Threshold effect observed: exceeding 7,000 steps shows measurable improvement
- Steps show stronger correlation with physical activity level (r = 0.77) than direct sleep impact
- May serve as accessible intervention target for sedentary populations

#### Sleep Disorder Prevalence

**Distribution:**
- No disorder: 58.6% of sample (219 participants)
- Sleep Apnea: 20.9% of sample (78 participants)
- Insomnia: 20.6% of sample (77 participants)
- Overall disorder prevalence: 41.4%

**Risk Factor Identification:**
- Elevated BMI demonstrates strong association with Sleep Apnea diagnosis
- High stress levels (≥7) correlate strongly with Insomnia presentation
- Occupational factors show significant variation: nurses (73 participants) and sales representatives demonstrate elevated disorder rates
- Age shows moderate correlation with sleep quality (r = 0.47)

#### Physiological Metrics

**Cardiovascular Indicators:**
- Mean heart rate: 70.17 beats per minute (within normal range)
- Standard deviation: 4.14 BPM
- Heart rate shows negative correlation with sleep quality (r = -0.66)
- Mean systolic blood pressure: 128.6 mmHg
- Mean diastolic blood pressure: 84.6 mmHg
- Blood pressure components show high intercorrelation (r = 0.97)

**Body Mass Index Distribution:**
- Normal/Normal Weight: 57.8% (216 participants)
- Overweight: 39.6% (148 participants)
- Obese: 2.7% (10 participants)

**Correlation Matrix Key Findings:**
- Strongest positive correlations: Sleep Duration-Quality (0.88), Physical Activity-Daily Steps (0.77), Systolic-Diastolic BP (0.97)
- Strongest negative correlations: Stress-Sleep Quality (-0.90), Stress-Sleep Duration (-0.81), Heart Rate-Sleep Quality (-0.66)

### Notable Research Findings

Several unexpected patterns emerged from the exploratory analysis:

1. **Age Factor:** Age demonstrates moderate predictive power for sleep quality in this cohort (r = 0.47), suggesting accumulated lifestyle factors or physiological changes affect sleep architecture with aging.

2. **Gender Differences:** Gender differences in sleep duration and quality scores are statistically insignificant, suggesting lifestyle factors supersede biological sex in this sample.

3. **Occupational Impact:** Occupation emerges as a significant predictor of sleep patterns, with shift work and high-stress professions showing particularly poor outcomes. Nurses constitute the largest occupational group (73 participants) with elevated stress and disorder rates.

4. **Heart Rate Paradox:** Elevated resting heart rate correlates with poor sleep quality (r = -0.66), potentially indicating heightened sympathetic nervous system activity or stress response.

5. **Daily Steps Independence:** Despite strong correlation between physical activity level and daily steps (r = 0.77), daily steps show minimal direct correlation with sleep quality (r = 0.02), suggesting intensity may matter more than volume.

## Data Preprocessing Methodology

### Data Cleaning Procedures

- Verification confirmed absence of missing values in critical columns
- Null values in Sleep Disorder column treated as distinct category ("None")
- Duplicate record check performed (none identified)
- Data type validation completed for all variables
- Outlier detection using Interquartile Range (IQR) method confirmed no anomalous values

### Feature Engineering

**Blood Pressure Decomposition:** Original 'Blood Pressure' variable separated into 'Systolic_BP' and 'Diastolic_BP' components to enable independent analysis of hypertension indicators.

**Derived Metrics Created:**
- **Sleep Efficiency:** Composite metric combining duration and quality: (Sleep Duration × Quality of Sleep) / 10
- **Activity-Stress Ratio:** Balance measure: Physical Activity Level / (Stress Level + 1)
- **Sleep Deficit:** Deviation from recommended duration: 8 - Sleep Duration

**Categorical Binning:** 
- Age groups created using clinically relevant boundaries: 20-30, 30-40, 40-50, 50-60, 60+ years
- Sleep duration categorized: Short (<6h), Normal (6-8h), Long (>8h)
- Physical activity levels classified into quartile-based categories: Low, Medium, High, Very High
- Stress categories: Low (1-3), Moderate (4-5), High (6-7), Very High (8-10)

### Encoding Strategies

**Label Encoding:** Applied to ordinal and nominal features using appropriate strategies:
- Gender: Binary encoding (Female=0, Male=1)
- BMI Category: Categorical encoding preserving distinct categories
- Occupation: Numerical codes assigned (0-10) for algorithmic processing
- Sleep Disorder: Categorical encoding (Insomnia=0, None=1, Sleep Apnea=2)

**Rationale:** This encoding approach preserves categorical distinctions while creating numerical representations suitable for machine learning algorithms.

### Normalization and Scaling

**Standardization (Z-score normalization):** StandardScaler applied to continuous variables to ensure unit-invariant analysis:
- Demographic: Age
- Sleep metrics: Sleep Duration, Quality of Sleep
- Lifestyle factors: Physical Activity Level, Stress Level, Daily Steps
- Physiological measures: Heart Rate, Systolic BP, Diastolic BP
- Engineered features: Sleep Efficiency, Activity-Stress Ratio, Sleep Deficit

**Min-Max Normalization:** Alternative scaling to [0,1] range created for algorithms sensitive to feature scales.

**Justification:** Standardization ensures all features operate on comparable scales, critical for distance-based algorithms and preventing feature dominance based solely on measurement units.

### Categorical Variable Treatment

- Sleep Disorder retained as target variable with three levels (None, Sleep Apnea, Insomnia)
- Original columns preserved for interpretability in visualization outputs
- Encoded versions created in separate datasets for modeling

### Data Validation

**Quality Control Measures:**
- Outlier detection using Interquartile Range (IQR) method
- Physiological range validation (Heart Rate: 65-86 bpm, Blood Pressure: clinically normal ranges)
- Logical consistency verification (e.g., sleep duration constrained to 5.8-8.5 hours)
- Cross-variable validation (e.g., Sleep Efficiency consistency checks)

**Generated Datasets:**
- **sleep_health_cleaned.csv:** Main dataset with engineered features (374 × 22 columns)
- **sleep_health_ml_ready.csv:** Fully encoded for machine learning (374 × 16 columns)
- **sleep_health_standardized.csv:** Standardized features for distance-based algorithms
- **sleep_health_normalized.csv:** Min-max normalized features for neural networks

## Machine Learning Implementation

### Sleep Quality Prediction Model

**Algorithm:** Random Forest Regressor  
**Framework:** scikit-learn (RandomForestRegressor)

**Model Configuration:**
- Number of estimators: 100 trees
- Random state: 42 (for reproducibility)
- Target variable: Quality of Sleep (continuous, 1-10 scale)
- Prediction type: Regression

**Feature Set (11 features):**
1. Age
2. Gender (encoded)
3. Occupation (encoded)
4. Sleep Duration
5. Physical Activity Level
6. Stress Level
7. BMI Category (encoded)
8. Heart Rate
9. Daily Steps
10. Systolic Blood Pressure
11. Diastolic Blood Pressure

**Model Architecture:**
- Ensemble method combining 100 decision trees
- Each tree trained on random subset of features
- Bootstrap aggregating (bagging) for variance reduction
- Default max_features for regression: all features
- No explicit max_depth constraint (trees grown until pure or minimum samples reached)

**Training Methodology:**
- Full dataset training (374 samples)
- Label encoding applied to categorical variables before training
- No train-test split in current implementation (model serves as predictor tool)
- Cached using Streamlit's @st.cache_resource for performance optimization

**Model Justification:**
Random Forest chosen for several key advantages:
1. Handles both numerical and encoded categorical features effectively
2. Resistant to overfitting through ensemble averaging
3. Captures non-linear relationships between lifestyle factors and sleep quality
4. Provides robust predictions without extensive hyperparameter tuning
5. Implicit feature importance ranking capability
6. No assumption of linear relationships (critical given strong non-linearities observed in EDA)

**Prediction Pipeline:**
1. User inputs collected via Streamlit interface
2. Categorical variables encoded using fitted LabelEncoders
3. Input features assembled into numpy array (1 × 11 shape)
4. Model prediction generated and constrained to [1, 10] range
5. Risk level classification based on predicted score:
   - Low Risk: ≥8.0
   - Moderate Risk: 6.0-7.9
   - High Risk: <6.0

**Model Limitations:**
- No formal validation set for performance metrics
- Trained on full dataset without cross-validation
- No confidence intervals or prediction uncertainty quantification
- Assumes independence of observations (no temporal or hierarchical structure)

**Planned Enhancements for Final Submission:**
- Cross-validation implementation for robust performance estimation
- Feature importance analysis and visualization
- Comparison with alternative algorithms (Gradient Boosting, Neural Networks)
- Hyperparameter optimization using GridSearchCV
- Prediction confidence intervals using quantile regression forests

## Streamlit Application Development

### Application Architecture

The interactive web application implements a multi-page architecture with the following components:

#### Home Page (app.py)
- Project overview and research objectives
- Summary statistics dashboard displaying key metrics
- Demographic distribution visualizations
- Sleep disorder prevalence breakdown
- Interactive tabs for Sleep Quality Distribution, Demographic Overview, and Health Indicators
- Quick insights section highlighting major findings
- Navigation guidance for application exploration

#### Data Explorer (1_Data_Explorer.py)
- Dynamic filtering interface with multiple criteria:
  - Age range slider
  - Gender selection
  - Multi-select occupation filter
  - Sleep disorder filter
  - BMI category filter
  - Sleep duration range
  - Quality of sleep range
- Real-time data table updates based on filter criteria
- Column selector for customized data views
- CSV export functionality for filtered datasets
- Three tabbed views:
  - Data Table: Filterable dataframe with gradient highlighting
  - Summary Statistics: Descriptive statistics for numerical and categorical features
  - Visual Overview: Scatter plots, box plots, and violin plots
- Correlation analysis with sleep quality for filtered subsets

#### Deep Dive Analysis (2_Deep_Dive_Analysis.py)
- Five comprehensive analysis modules:
  - **Correlation Analysis:** Full correlation matrix heatmap, key correlations with sleep quality, scatter matrix for selected features
  - **Demographic Trends:** Age group analysis, gender comparisons, statistical significance testing (t-tests, ANOVA)
  - **Occupation Analysis:** Sleep quality rankings by profession, stress level comparisons, comprehensive multi-metric visualization
  - **Health Indicators:** BMI impact analysis, blood pressure relationships, physical activity correlations, daily steps categorization
  - **Sleep Disorder Profiles:** Distribution analysis, characteristic comparisons using radar charts, detailed metric distributions, occupation-disorder cross-tabulation

#### Sleep Quality Predictor (3_Sleep_Quality_Predictor.py)
- Comprehensive input form collecting:
  - Demographics: Age, gender, occupation, BMI category
  - Sleep patterns: Duration
  - Lifestyle factors: Physical activity level, daily steps, stress level
  - Physiological metrics: Heart rate, blood pressure (systolic/diastolic)
- Random Forest prediction engine generating quality scores
- Results display including:
  - Predicted quality score with visual gauge
  - Risk level assessment with color-coded messaging
  - Comparison to population average
  - Gauge chart visualization with reference threshold
- Personalized recommendation system based on input parameters:
  - Sleep duration optimization
  - Physical activity suggestions
  - Stress management strategies
  - Daily steps targets
  - Cardiovascular health monitoring
  - Weight management guidance
- Comparison visualization: User metrics vs dataset averages

#### Insights Dashboard (4_Insights_Dashboard.py)
- Executive summary with four key metrics cards
- Top 5 findings presentation with supporting evidence
- Three tabbed visual insight sections:
  - **Lifestyle Factors:** Physical activity impact, stress level categories, daily steps analysis
  - **Demographics & Health:** Age group radar charts, BMI-disorder prevalence, gender comparison
  - **Sleep Disorders:** Distribution analysis, characteristic comparisons, risk factor box plots, occupation-disorder relationships
- Interactive summary tool: Risk profile simulator with adjustable sliders for real-time prediction updates
- Factor contribution analysis showing impact of each variable on sleep quality

### Technical Implementation

**Performance Optimization:**
- Implementation of Streamlit's caching decorators:
  - @st.cache_data for data loading and preprocessing operations
  - @st.cache_resource for model training (prevents retraining on every interaction)
- Efficient pandas vectorized operations throughout analysis pipeline
- Lazy loading strategies for computationally intensive visualizations
- Optimized data structures for filtering operations

**Visualization Framework:**
- **Plotly:** Primary library for interactive visualizations with zoom, pan, and hover capabilities
  - Graph Objects API for custom chart configurations
  - Express API for rapid exploratory visualizations
  - Dark theme template applied consistently across all plots
- **Seaborn/Matplotlib:** Used for correlation heatmaps with gradient color schemes
- **Plotly Subplots:** Multi-panel comparative visualizations
- Custom color schemes aligned with dark theme aesthetic

**User Experience Design:**
- Consistent dark theme (background: #0E1117, cards: #1A1D24)
- Responsive column layouts adapting to screen dimensions
- Tab-based organization for content hierarchy
- Metric cards for key performance indicators with large numerical displays
- Sidebar navigation for intuitive page transitions
- Loading indicators for asynchronous operations
- Color-coded risk levels (green: low, orange: moderate, red: high)
- Gradient accent cards for important information highlighting
- Custom CSS styling for professional interface appearance

**Interactive Features:**
- Real-time filtering and data updates
- Dynamic chart generation based on user selections
- Hover tooltips displaying detailed observation information
- Download functionality for filtered datasets
- Multi-select filters with "All" option support
- Slider controls for continuous variable filtering
- Risk profile simulator with instant prediction updates

**Theme Consistency:**
- Custom CSS file (assets/style.css) enforcing dark theme across all components
- Consistent color palette: Primary blue (#3498DB), Success green (#2ECC71), Warning orange (#F39C12), Danger red (#E74C3C)
- Typography: Helvetica Neue font family throughout
- Border and shadow effects for depth perception
- Smooth transitions and hover effects

## Project Structure

```
sleep-health-analysis/
│
├── data/
│   ├── sleep_health_cleaned.csv
│   ├── sleep_health_ml_ready.csv
│   ├── sleep_health_standardized.csv
│   ├── sleep_health_normalized.csv
│   ├── summary_statistics.json
│   ├── correlation_matrices.json
│   ├── feature_statistics.json
│   ├── encoder_mappings.json
│   └── data_dictionary.json
│
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Deep_Dive_Analysis.py
│   ├── 3_Sleep_Quality_Predictor.py
│   └── 4_Insights_Dashboard.py
│
├── assets/
│   └── style.css
│
├── app.py
├── requirements.txt
└── README.md
```

## Technology Stack

- **Python 3.9+:** Primary programming language
- **Streamlit 1.x:** Web application framework for interactive dashboards
- **Pandas 2.x:** Data manipulation and analysis
- **NumPy:** Numerical computation and array operations
- **Plotly 5.x:** Interactive data visualization
- **Scikit-learn:** Machine learning (Random Forest), preprocessing, and encoding
- **SciPy:** Statistical testing (t-tests, ANOVA)
- **Seaborn/Matplotlib:** Statistical visualization and heatmaps

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

### Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
scipy>=1.11.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

## Live Application

**Production URL:** https://sleep-health-analysis-yashwitha.streamlit.app/

**Platform:** Streamlit Cloud  
**Status:** Active

## Future Development Roadmap

### Planned Enhancements for Final Submission

**Primary Addition: Conversational AI Chatbot**
- Interactive Q&A interface for sleep health queries
- Natural language understanding of user questions
- Contextual responses based on dataset insights
- Personalized recommendations through conversational flow
- Integration with existing prediction models
- Memory of conversation history for contextual awareness

**Additional Machine Learning Enhancements:**
- Implementation of additional supervised learning models (Gradient Boosting, XGBoost)
- Cross-validation and comprehensive performance metric reporting (RMSE, MAE, R²)
- Feature importance analysis and visualization
- Prediction confidence intervals using quantile regression
- Sleep disorder classification models (multi-class classification)
- Model comparison framework with performance benchmarking

**Advanced Visualization Techniques:**
- 3D scatter plot projections for multi-variable relationships
- Network graphs for feature interaction visualization
- Time-series analysis capabilities for longitudinal data
- Interactive correlation explorer with drill-down capabilities
- Animated transitions for temporal comparisons

**User Experience Improvements:**
- User data upload and analysis functionality
- Comparative analysis tool for similar demographic profiles
- PDF report generation capability with comprehensive insights
- Export functionality for personalized recommendations
- Dashboard customization options
- Mobile-responsive design optimization

**Educational Content:**
- Educational content modules on sleep science
- Embedded tooltips explaining statistical concepts
- Glossary of sleep health terminology
- Links to peer-reviewed research
- Best practices guide for sleep hygiene

### Extended Features (Post-Course)

- Temporal analysis capabilities for longitudinal data tracking
- Unsupervised clustering for sleep profile identification and segmentation
- Experimental design framework for lifestyle intervention studies
- Wearable device data integration (Fitbit, Apple Watch, etc.)
- Social sharing capabilities for insights
- Multi-language support for broader accessibility
- API endpoint creation for programmatic access

## Academic Learnings

### Technical Competencies Developed

1. **End-to-End Data Science Pipeline:** Complete workflow from raw data acquisition through preprocessing, analysis, modeling, and deployment
2. **Interactive Web Application Development:** Proficiency in Streamlit framework including multi-page architecture, state management, and caching strategies
3. **Advanced Data Visualization:** Mastery of interactive visualization libraries (Plotly) and effective visual communication of complex statistical relationships
4. **Data Preprocessing Methodologies:** Comprehensive understanding of encoding strategies, feature scaling, feature engineering, and data validation techniques
5. **Machine Learning Implementation:** Practical experience with ensemble methods (Random Forest), model training, and prediction pipeline development
6. **Statistical Analysis:** Application of correlation analysis, hypothesis testing (t-tests, ANOVA), and distribution analysis
7. **User-Centered Design:** Development of intuitive interfaces for non-technical audiences with emphasis on accessibility and usability
8. **Version Control and Documentation:** Best practices in code organization, documentation, and technical writing

### Domain Knowledge Acquired

1. **Sleep Science Fundamentals:** Understanding of sleep architecture, sleep quality metrics, and factors affecting sleep health
2. **Health Data Analysis:** Methodologies for analyzing physiological metrics, cardiovascular indicators, and lifestyle factors
3. **Behavioral Pattern Analysis:** Insights into relationships between daily habits, occupational factors, and health outcomes
4. **Risk Assessment Frameworks:** Development of evidence-based risk classification systems for health screening
5. **Public Health Considerations:** Understanding of sleep disorder prevalence, demographic variations, and intervention strategies
6. **Psychophysiology:** Connections between stress, autonomic nervous system activity (heart rate), and sleep quality
7. **Clinical Guidelines:** Familiarity with evidence-based recommendations for sleep duration, physical activity, and cardiovascular health

### Research Methodology Skills

1. **Exploratory Data Analysis:** Systematic investigation of data structure, distributions, and relationships
2. **Hypothesis Generation:** Identifying research questions from data patterns
3. **Evidence Synthesis:** Combining multiple analytical approaches to support conclusions
4. **Results Communication:** Translating technical findings into accessible insights
5. **Critical Evaluation:** Assessing limitations, biases, and generalizability of findings

## Key Insights and Takeaways

### Major Findings

1. **Stress as Primary Determinant:** Stress level emerged as the strongest predictor of sleep quality (r = -0.90), surpassing all other lifestyle and physiological factors. This finding emphasizes the critical importance of stress management interventions.

2. **Duration-Quality Coupling:** Sleep duration and quality are tightly coupled (r = 0.88), suggesting interventions targeting one dimension will likely benefit the other.

3. **Occupational Disparities:** Significant variation in sleep health across professions, with healthcare workers and sales professionals showing elevated risk profiles. This suggests need for occupation-specific interventions.

4. **Activity Benefits Plateau:** While physical activity shows positive effects, the relationship appears to plateau around 60 minutes per day, suggesting diminishing returns beyond moderate activity levels.

5. **Cardiovascular-Sleep Connection:** Elevated resting heart rate correlates with poor sleep quality, potentially indicating shared underlying stress or autonomic dysfunction.

### Practical Applications

1. **Screening Tool:** The prediction model provides a non-invasive screening tool for identifying individuals at risk for sleep quality issues.

2. **Personalized Recommendations:** Individual-level analysis enables tailored lifestyle modification suggestions based on specific risk profiles.

3. **Population Health Insights:** Occupational and demographic patterns inform targeted public health interventions.

4. **Intervention Prioritization:** Correlation strengths guide prioritization of intervention targets (stress management > physical activity > sleep hygiene).

## References

**Dataset Source:**  
Sleep Health and Lifestyle Dataset. Kaggle.  
Available at: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

**Technical Documentation:**
- Streamlit Documentation: https://docs.streamlit.io
- Plotly Python Documentation: https://plotly.com/python
- Scikit-learn Documentation: https://scikit-learn.org

**Sleep Science Resources:**
- National Sleep Foundation Clinical Guidelines
- Centers for Disease Control and Prevention: Sleep and Sleep Disorders
- American Academy of Sleep Medicine: Clinical Practice Guidelines

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
- Machine learning model selection and implementation
- Documentation formatting and academic writing conventions
- Code optimization and debugging support

Claude served as a technical consultation resource similar to office hours or tutoring support. All conceptual framework decisions, data analysis interpretations, research insights, and academic content were developed independently by the author.

## License

This project is submitted as coursework for academic evaluation. All rights reserved. The dataset is used under Kaggle's terms of service for educational purposes.

---

**Project Status:** In Progress - Midterm Submission  
**Last Updated:** October 2025

**Note:** This project represents work completed through the midterm evaluation period. Additional features including a conversational AI chatbot, enhanced machine learning models, and advanced analytical capabilities are planned for final submission.
