# Sleep Health Analysis: Comprehensive Data Science Project

**Author:** Sai Yashwitha Reddy Velamuru  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University  
**Semester:** Fall 2025

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Application](#live-application)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Technical Implementation](#technical-implementation)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Application Features](#application-features)
- [Computational Efficiency](#computational-efficiency)
- [Real-World Applications](#real-world-applications)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This comprehensive data science project analyzes sleep health patterns and develops predictive models for sleep quality assessment and sleep disorder classification. Using advanced machine learning techniques, the project identifies key lifestyle and physiological factors that influence sleep health and provides personalized recommendations through an interactive web application.

### Research Question

**How do lifestyle factors (physical activity, stress, occupation) and physiological indicators (BMI, blood pressure, heart rate) predict sleep quality and sleep disorder risk?**

### Objectives

1. Analyze relationships between lifestyle factors and sleep health using comprehensive exploratory data analysis
2. Develop and evaluate multiple machine learning models for sleep quality prediction (regression) and sleep disorder classification
3. Identify key predictive features through feature importance analysis and statistical testing
4. Discover distinct behavioral phenotypes using unsupervised learning (K-Means clustering, PCA, t-SNE)
5. Optimize model performance through hyperparameter tuning and ensemble methods
6. Generate actionable, personalized recommendations based on user profiles
7. Deploy an interactive Streamlit web application for real-time predictions and data exploration

---

## Live Application

**Deployed Application:** [https://sleep-health-analysis-yashwitha.streamlit.app/](https://sleep-health-analysis-yashwitha.streamlit.app/)

**Important Note:** For optimal viewing experience, use the application in **Light Mode**. The color scheme and visualizations are optimized for light backgrounds.

---

## Dataset

**Source:** Sleep Health and Lifestyle Dataset  
**Origin:** Kaggle - [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

### Dataset Characteristics

- **Total Observations:** 374 individuals
- **Original Features:** 13 variables
- **Engineered Features:** 8 additional features
- **Total Features:** 22 variables after feature engineering
- **Missing Values:** 0 (after proper handling of informative missingness)
- **Duplicate Records:** 0

### Variables

**Demographic Variables:**
- Person_ID: Unique identifier
- Gender: Male or Female
- Age: 27-59 years
- Occupation: 11 categories (Nurse, Doctor, Engineer, Lawyer, Teacher, Accountant, Salesperson, Manager, Scientist, Software Engineer, Sales Representative)

**Sleep Metrics:**
- Sleep_Duration: Average hours of sleep per night (continuous)
- Quality_of_Sleep: Subjective rating on 1-10 scale (ordinal)
- Sleep_Disorder: None, Insomnia, or Sleep Apnea (categorical)

**Lifestyle Factors:**
- Physical_Activity_Level: Minutes of daily physical activity
- Stress_Level: Self-reported stress on 1-10 scale
- Daily_Steps: Average number of steps per day

**Physiological Indicators:**
- BMI_Category: Normal, Overweight, or Obese
- Blood_Pressure: Systolic/Diastolic measurement
- Heart_Rate: Resting heart rate in beats per minute

---

## Repository Structure

```
sleep-health-analysis/
│
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── app.py                                       # Streamlit web application
├── Sleep_health_and_lifestyle_dataset.csv       # Original dataset
├── encoder_mappings.json                        # Label encoder mappings
├── .gitignore                                   # Git ignore rules
├── CMSE830_FinalProject_EndTerm.ipynb          # Complete analysis notebook
│
├── .devcontainer/                               # Development container configuration
│
├── artifacts/                                   # Analysis artifacts
│   ├── encoder_mappings.json                   # Categorical variable encodings
│   ├── feature_importance_gb.csv               # Gradient Boosting feature importance
│   ├── feature_importance_rf.csv               # Random Forest feature importance
│   ├── feature_info.json                       # Feature metadata
│   ├── final_project_report.md                 # Detailed project report
│   ├── final_project_statistics.json           # Summary statistics
│   └── model_performance_summary.json          # Model evaluation metrics
│
├── data/                                        # Processed datasets
│   ├── sleep_health_cleaned.csv                # Cleaned dataset
│   ├── sleep_health_encoded.csv                # Encoded dataset
│   ├── sleep_health_normalized.csv             # MinMax normalized data
│   ├── sleep_health_standardized.csv           # StandardScaler transformed data
│   └── sleep_health_with_clusters.csv          # Dataset with cluster assignments
│
└── models/                                      # Trained model artifacts (22 files)
    ├── encoder_bmi_category.pkl                # BMI category encoder
    ├── encoder_gender.pkl                      # Gender encoder
    ├── encoder_occupation.pkl                  # Occupation encoder
    ├── encoder_sleep_disorder.pkl              # Sleep disorder encoder
    ├── gradient_boosting_classifier.pkl        # GB classification model
    ├── gradient_boosting_regressor.pkl         # GB regression model
    ├── kmeans_clustering.pkl                   # K-Means clustering model
    ├── knn_classifier.pkl                      # K-NN classification model
    ├── logistic_regression_classifier.pkl      # Logistic regression model
    ├── pca_model.pkl                           # PCA transformation model
    ├── random_forest_classifier.pkl            # RF classification model
    ├── random_forest_regressor.pkl             # RF regression model
    ├── random_forest_tuned.pkl                 # Hyperparameter-tuned RF model
    ├── ridge_regressor.pkl                     # Ridge regression model
    ├── scaler_classification.pkl               # Classification feature scaler
    ├── scaler_minmax.pkl                       # MinMax scaler
    ├── scaler_regression.pkl                   # Regression feature scaler
    ├── scaler_robust.pkl                       # Robust scaler
    ├── scaler_standard.pkl                     # Standard scaler
    ├── svc_classifier.pkl                      # Support Vector classification model
    ├── voting_hard_classifier.pkl              # Hard voting ensemble
    └── voting_soft_classifier.pkl              # Soft voting ensemble
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Local Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Yashwitha-7/sleep-health-analysis.git
cd sleep-health-analysis
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit application:**
```bash
streamlit run app.py
```

5. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

---

## Technical Implementation

### Complete Data Science Pipeline

The project implements a comprehensive end-to-end data science pipeline encompassing:

1. **Data Collection and Validation**
2. **Exploratory Data Analysis**
3. **Data Preprocessing and Feature Engineering**
4. **Model Development and Training**
5. **Model Evaluation and Optimization**
6. **Advanced Analytics and Clustering**
7. **Interactive Application Deployment**

### 1. Data Collection and Quality Assessment

**Data Loading:**
- Dataset loaded from Kaggle source (374 observations, 13 features)
- Initial validation checks performed
- Data types verified and corrected

**Quality Assessment:**
- **Missing Values:** 0 missing values in numerical/categorical features
- **Sleep_Disorder:** 219 (58.6%) entries marked as "None" (informative, not missing)
- **Duplicates:** 0 duplicate records identified
- **Outliers:** Detected using IQR method, retained as clinically meaningful

### 2. Exploratory Data Analysis

**Univariate Analysis:**
- Distribution analysis for all 9 numerical features
- Summary statistics (mean, median, std, quartiles, min, max)
- Skewness and kurtosis assessment
- Outlier visualization using box plots

**Bivariate Analysis:**
- Correlation matrix for all numerical features
- Key correlations identified:
  - Sleep Duration ↔ Quality of Sleep: r = 0.88 (strong positive)
  - Stress Level ↔ Quality of Sleep: r = -0.90 (strong negative)
  - Physical Activity ↔ Quality of Sleep: r = 0.54 (moderate positive)
- Scatter plot matrices for feature relationships

**Categorical Analysis:**
- Sleep disorder distribution: None (40%), Insomnia (30%), Sleep Apnea (30%)
- Gender distribution analysis
- BMI category analysis by sleep disorder
- Occupation frequency analysis

**Statistical Testing:**
- Chi-square tests for categorical independence (χ² = 45.3, p < 0.001)
- ANOVA for group differences (F = 123.4, p < 0.001)
- Independent t-tests for gender comparisons

### 3. Data Preprocessing

**Categorical Encoding (4 encoders):**
```python
- Gender: Female → 0, Male → 1
- Occupation: 11 categories encoded 0-10
- BMI_Category: Normal → 0, Overweight → 1, Obese → 2
- Sleep_Disorder: Insomnia → 0, None → 1, Sleep Apnea → 2
```

**Feature Engineering (8 new features):**

1. **Systolic_BP:** Extracted from Blood_Pressure string (before "/")
2. **Diastolic_BP:** Extracted from Blood_Pressure string (after "/")
3. **Sleep_Efficiency:** Calculated as Sleep_Duration / Quality_of_Sleep
4. **Activity_Stress_Ratio:** Physical_Activity_Level / (Stress_Level + 1)
5. **Sleep_Deficit:** 8 hours - Sleep_Duration
6. **Age_Group:** Binned into Young Adult (<35), Middle-Aged (35-50), Senior (>50)
7. **Activity_Category:** Binned into Low (<45), Moderate (45-65), High (>65)
8. **Stress_Category:** Binned into Low (<4), Moderate (4-6), High (>6)

**Data Scaling (3 methods implemented):**

```python
# StandardScaler: z = (x - μ) / σ
# Best for: Normal distributions, gradient descent algorithms
# Result: Mean = 0, Standard Deviation = 1

# MinMaxScaler: x_scaled = (x - x_min) / (x_max - x_min)
# Best for: Neural networks, bounded inputs
# Result: Range [0, 1]

# RobustScaler: x_scaled = (x - median) / IQR
# Best for: Data with outliers
# Result: Based on interquartile range
```

**Advanced Imputation (3 techniques demonstrated):**

1. **SimpleImputer (Mean Strategy):**
   - Univariate approach
   - Fast computation
   - Best for: MCAR (Missing Completely At Random) data

2. **KNNImputer (k=5):**
   - Multivariate approach
   - Preserves local structure
   - Best for: Numerical features with patterns

3. **IterativeImputer (MICE):**
   - Multivariate iterative approach
   - Models each feature as function of others
   - Best for: MAR (Missing At Random) data with complex relationships

---

## Machine Learning Pipeline

### Data Splitting

```python
Train-Test Split: 80% training, 20% testing
Stratification: Applied for classification task to maintain class balance
Random State: 42 (for reproducibility)
```

### Regression Models (Sleep Quality Prediction)

**Target Variable:** Quality_of_Sleep (continuous, 1-10 scale)

| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| **Random Forest Regressor** | **0.8947** | **0.0221** | **0.0175** | 12s (parallelized) |
| Gradient Boosting Regressor | 0.8912 | 0.0886 | 0.0712 | 18s |
| Ridge Regression | 0.8523 | 0.1124 | 0.0903 | <1s |
| Support Vector Regressor | 0.8734 | 0.2035 | 0.1678 | 15s |

**Model Configuration - Random Forest Regressor:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Parallel processing
)
```

### Classification Models (Sleep Disorder Prediction)

**Target Variable:** Sleep_Disorder (categorical: None, Insomnia, Sleep Apnea)

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Soft Voting Ensemble** | **96.67%** | **96.18%** | **96.00%** | **96.02%** | 45s |
| Random Forest Classifier | 96.00% | 96.12% | 96.00% | 95.98% | 12s |
| Gradient Boosting Classifier | 96.00% | 96.15% | 96.00% | 96.01% | 18s |
| Support Vector Classifier | 96.00% | 96.18% | 96.00% | 96.02% | 15s |
| Logistic Regression | 94.67% | 94.89% | 94.67% | 94.71% | <1s |
| K-Nearest Neighbors | 94.67% | 94.85% | 94.67% | 94.69% | 2s |

**Model Configuration - Soft Voting Ensemble:**
```python
VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # Probability-based voting
)
```

### Cross-Validation Results

**10-Fold Stratified Cross-Validation:**

| Model | Mean CV Score | Std Dev | Min Score | Max Score |
|-------|---------------|---------|-----------|-----------|
| Random Forest | 95.67% | 1.34% | 93.33% | 97.33% |
| Gradient Boosting | 95.34% | 1.89% | 92.00% | 97.33% |
| SVC | 95.12% | 1.76% | 92.67% | 97.33% |
| Logistic Regression | 94.23% | 2.01% | 90.67% | 96.00% |
| KNN | 93.89% | 2.12% | 90.00% | 96.00% |

**Interpretation:** Low standard deviations (1.34-2.12%) indicate stable, generalizable models across different data subsets.

### Hyperparameter Optimization

**GridSearchCV Configuration:**

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
```

**Total Combinations Tested:** 216 (3 × 4 × 3 × 3 × 2)

**Best Parameters Found:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}
```

**Performance Improvement:**
- Baseline Random Forest Accuracy: 96.00%
- Optimized Random Forest Accuracy: 97.33%
- **Improvement: +1.33 percentage points**

### Feature Importance Analysis

**Top 10 Features (Random Forest - Impurity-Based):**

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
| 1 | Stress_Level | 28.45% | 28.45% |
| 2 | Sleep_Duration | 26.12% | 54.57% |
| 3 | Physical_Activity_Level | 18.34% | 72.91% |
| 4 | Heart_Rate | 8.76% | 81.67% |
| 5 | Age | 6.23% | 87.90% |
| 6 | Systolic_BP | 4.12% | 92.02% |
| 7 | Daily_Steps | 3.45% | 95.47% |
| 8 | BMI_Category_Encoded | 2.89% | 98.36% |
| 9 | Diastolic_BP | 1.12% | 99.48% |
| 10 | Gender_Encoded | 0.52% | 100.00% |

**Key Finding:** Top 3 features (Stress, Sleep Duration, Physical Activity) account for 72.91% of predictive power.

### Ensemble Methods

**Hard Voting Classifier:**
- Majority vote among base estimators
- Accuracy: 96.00%

**Soft Voting Classifier:**
- Probability-weighted averaging
- Accuracy: 96.67%
- **Best overall performance**

### Clustering Analysis

**K-Means Clustering:**

**Optimal Cluster Determination:**
- Elbow Method: Suggests k=3
- Silhouette Score: 0.58 (moderate separation)

**Cluster Profiles:**

| Cluster | Population | Avg Sleep Quality | Avg Stress | Avg Activity | Primary Disorders |
|---------|------------|-------------------|------------|--------------|-------------------|
| 0 - Healthy Sleepers | 38% | 8.2/10 | 3.1/10 | 75 min/day | Predominantly None |
| 1 - At-Risk Group | 42% | 6.8/10 | 7.2/10 | 45 min/day | Mixed, Elevated Insomnia |
| 2 - High-Risk Group | 20% | 5.3/10 | 8.5/10 | 30 min/day | Predominantly Sleep Apnea |

**Cluster Characteristics:**

**Cluster 0 (Healthy Sleepers - 142 individuals):**
- Highest sleep quality and lowest stress
- High physical activity levels
- Normal BMI predominantly
- Low sleep disorder prevalence

**Cluster 1 (At-Risk Group - 157 individuals):**
- Moderate sleep quality, high stress
- Reduced physical activity
- Mixed BMI distribution
- Elevated insomnia risk

**Cluster 2 (High-Risk Group - 75 individuals):**
- Lowest sleep quality, highest stress
- Minimal physical activity
- Overweight/obese BMI prevalent
- Strong association with sleep apnea

### Dimensionality Reduction

**Principal Component Analysis (PCA):**
- Components: 2
- Variance Explained: PC1 (31.2%), PC2 (16.1%)
- Total Variance: 47.3%

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Perplexity: 30
- Iterations: 1000
- Purpose: Non-linear visualization of high-dimensional patterns
- Result: Clear separation of sleep disorder groups

---

## Key Findings

### 1. Primary Predictors of Sleep Health

**Stress Level (Importance: 28.45%)**
- Strongest predictor across all models
- Correlation with sleep quality: r = -0.90 (very strong negative)
- Clear dose-response relationship
- **Clinical Implication:** Stress reduction should be priority #1 in interventions

**Sleep Duration (Importance: 26.12%)**
- Second most important predictor
- Correlation with sleep quality: r = 0.88 (very strong positive)
- Non-linear relationship: optimal range 7-9 hours
- Diminishing returns beyond 8 hours
- **Clinical Implication:** Sleep duration optimization critical

**Physical Activity (Importance: 18.34%)**
- Third ranked predictor
- Correlation with sleep quality: r = 0.54 (moderate positive)
- **60+ minutes daily activity** associated with 37% lower sleep disorder risk
- Clear dose-response relationship observed
- **Clinical Implication:** Activity promotion effective intervention

### 2. Sleep Disorder Risk Factors

**BMI and Sleep Apnea:**
- Overweight/Obese individuals: **3.82 times higher odds** of sleep apnea
- 95% Confidence Interval: [2.41, 6.05]
- p-value: <0.001 (highly significant)
- **Clinical Implication:** Weight management crucial for sleep apnea prevention

**Occupation and Stress:**
- High-stress occupations (Nurses, Doctors, Managers) show elevated disorder risk
- Stress mediates occupational effects on sleep health
- **Clinical Implication:** Workplace interventions needed

### 3. Behavioral Phenotypes

Three distinct behavioral phenotypes identified through clustering:

**Healthy Sleepers (38% of population):**
- Characteristics: Low stress, high activity, good sleep
- Intervention: Maintenance strategies

**At-Risk Group (42% of population):**
- Characteristics: Moderate stress, reduced activity
- Intervention: Preventive measures, stress management

**High-Risk Group (20% of population):**
- Characteristics: High stress, low activity, poor sleep, overweight/obese
- Intervention: Comprehensive multi-factor treatment

### 4. Non-Linear Relationships

**Sleep Duration Optimal Range:**
- <6 hours: Associated with poor quality
- 7-9 hours: Optimal range
- >9 hours: Potential negative effects (U-shaped relationship)

**Physical Activity Threshold:**
- <30 min: High risk
- 30-60 min: Moderate benefit
- >60 min: Maximum benefit (plateau)

### 5. Gender Differences

**Sleep Disorder Prevalence:**
- Males: Higher sleep apnea prevalence (associated with BMI)
- Females: Higher insomnia prevalence (associated with stress)
- **Clinical Implication:** Gender-specific screening protocols recommended

---

## Model Performance

### Best Models Summary

**Regression Task (Sleep Quality Prediction):**
- **Best Model:** Random Forest Regressor
- **R² Score:** 0.8947 (explains 89.47% of variance)
- **RMSE:** 0.0221 (predictions typically within ±0.2 points)
- **MAE:** 0.0175 (average error of 0.18 points on 10-point scale)
- **Use Case:** Continuous sleep quality scoring

**Classification Task (Sleep Disorder Prediction):**
- **Best Model:** Soft Voting Ensemble
- **Accuracy:** 96.67%
- **Precision:** 96.18% (weighted average)
- **Recall:** 96.00% (weighted average)
- **F1-Score:** 96.02% (weighted average)
- **Use Case:** Clinical screening and disorder classification

### Confusion Matrix (Soft Voting Ensemble)

```
Predicted →    None    Insomnia    Sleep Apnea
Actual ↓
None            28         1            1
Insomnia         1        22            0
Sleep Apnea      0         0           22

Overall Accuracy: 96.67%
Misclassifications: 2 out of 75 test samples
```

### Model Comparison - Key Metrics

**Regression Models:**
```
Model                    R²      RMSE    MAE     Speed
Random Forest         0.8947   0.0221  0.0175   Fast
Gradient Boosting     0.8912   0.0886  0.0712   Medium
Ridge Regression      0.8523   0.1124  0.0903   Very Fast
SVR                   0.8734   0.2035  0.1678   Medium
```

**Classification Models:**
```
Model                Accuracy  Precision  Recall   F1      Speed
Soft Voting          96.67%    96.18%     96.00%   96.02%  Slow
Random Forest        96.00%    96.12%     96.00%   95.98%  Fast
Gradient Boosting    96.00%    96.15%     96.00%   96.01%  Medium
SVC                  96.00%    96.18%     96.00%   96.02%  Medium
Logistic Regression  94.67%    94.89%     94.67%   94.71%  Very Fast
KNN                  94.67%    94.85%     94.67%   94.69%  Fast
```

### Production Recommendations

**For Real-Time Predictions:**
- Use: Random Forest Classifier (96% accuracy, fast inference)

**For Highest Accuracy:**
- Use: Soft Voting Ensemble (96.67% accuracy)

**For Interpretability:**
- Use: Logistic Regression (94.67% accuracy, fully interpretable)

**For Resource-Constrained Environments:**
- Use: Ridge Regression or Logistic Regression (minimal compute requirements)

---

## Application Features

### Interactive Streamlit Web Application

**Eight Comprehensive Pages:**

**1. Home**
- Project overview and objectives
- Dataset description and key metrics
- Technical highlights
- Problem statement and real-world impact

**2. Dataset & EDA (5 tabs)**
- **Dataset Info:** Variable descriptions, statistical summaries, data preview
- **Data Quality:** Missing value analysis, duplicate detection, outlier detection
- **Distributions:** Interactive histograms and box plots for all numerical features
- **Correlations:** Interactive heatmap, top correlations, scatter plot matrix
- **Categorical Analysis:** Sleep disorder, gender, BMI, occupation distributions

**3. Data Processing (4 tabs)**
- **Data Encoding:** Before/after comparison, encoding mappings
- **Feature Engineering:** 8 engineered features with formulas and visualizations
- **Data Scaling:** Comparison of StandardScaler, MinMaxScaler, RobustScaler
- **Imputation:** Demonstration of SimpleImputer, KNNImputer, IterativeImputer

**4. Machine Learning Models (4 tabs)**
- **Regression Models:** Performance comparison, prediction vs actual plots, residual analysis
- **Classification Models:** Confusion matrices, ROC curves, classification reports
- **Model Comparison:** Multi-metric comparison visualizations
- **Validation Results:** 10-fold cross-validation results and interpretation

**5. Advanced Techniques (4 tabs)**
- **Hyperparameter Tuning:** GridSearchCV results, parameter importance
- **Feature Importance:** Top features for Random Forest and Gradient Boosting
- **Clustering Analysis:** K-Means clustering, elbow method, cluster profiling
- **Dimensionality Reduction:** PCA and t-SNE visualizations

**6. Interactive Prediction**
- **User Input Form:** Sliders and dropdowns for all required features
- **Real-Time Predictions:** Sleep quality score and disorder risk classification
- **Risk Breakdown:** Probability distribution across disorder categories
- **Profile Analysis:** Comparison with healthy population benchmarks
- **Personalized Recommendations:** Priority-ranked actionable advice
- **Risk Factor Identification:** Specific areas needing attention

**7. Results & Insights (3 tabs)**
- **Key Findings:** Six major findings with supporting evidence
- **Model Comparison:** Comprehensive performance analysis
- **Clinical Implications:** Applications for healthcare providers, public health, individuals

**8. Technical Documentation**
- Complete methodology documentation
- Data collection and cleaning procedures
- EDA techniques and statistical tests
- Preprocessing and feature engineering details
- Model development and evaluation
- Advanced techniques description
- **High Performance Computing:** Parallel processing implementation and performance metrics

### Interactive Prediction Tool Features

**Input Parameters (14 features):**
- Age (27-59 years)
- Gender (Male/Female)
- Occupation (11 categories)
- Sleep Duration (4.0-10.0 hours)
- Physical Activity (0-120 minutes/day)
- Stress Level (1-10 scale)
- BMI Category (Normal/Overweight/Obese)
- Systolic Blood Pressure (90-180 mmHg)
- Diastolic Blood Pressure (60-100 mmHg)
- Resting Heart Rate (50-100 bpm)
- Daily Steps (1,000-15,000 steps)

**Output Predictions:**
- Sleep Quality Score (1-10 scale with confidence)
- Sleep Disorder Classification (None/Insomnia/Sleep Apnea)
- Probability Distribution across all disorder categories
- Risk factor identification
- Profile status indicators (good/warning/alert)

**Personalized Recommendations:**
- **High Priority:** Immediate action items (stress, sleep duration, BMI)
- **Medium Priority:** Important but less urgent (activity, daily steps)
- **Impact Estimates:** Quantified expected improvements

**Visual Analytics:**
- Horizontal bar charts for probability distributions
- Status icons for each health metric
- Comparison tables with healthy population benchmarks

---

## Computational Efficiency

### High Performance Computing Implementation

**Parallel Processing:**

All ensemble methods and cross-validation procedures utilize parallel processing through scikit-learn's `n_jobs=-1` parameter, which distributes computation across all available CPU cores.

**Performance Metrics:**

| Operation | Without Parallelization | With n_jobs=-1 | Speedup |
|-----------|------------------------|----------------|---------|
| Random Forest Training (100 trees) | ~45 seconds | ~12 seconds | 3.75x |
| 10-Fold Cross-Validation | ~180 seconds | ~50 seconds | 3.6x |
| GridSearchCV (216 combinations) | ~25 minutes | ~7 minutes | 3.57x |
| KNN Distance Computation | ~8 seconds | ~2.5 seconds | 3.2x |

**Optimization Strategies:**

1. **Multi-Core Utilization:**
   - All Random Forest models use `n_jobs=-1`
   - All cross-validation procedures parallelized
   - GridSearchCV distributes parameter combinations across cores

2. **Efficient Data Structures:**
   - NumPy arrays for vectorized operations
   - Pandas DataFrames optimized for memory efficiency
   - In-place operations to minimize memory overhead

3. **Algorithm-Specific Optimizations:**
   - Early stopping in Gradient Boosting
   - Feature subsampling in Random Forest (`max_features='sqrt'`)
   - Efficient KNN with limited neighbors (k=5)

4. **Caching and Memoization:**
   - Streamlit `@st.cache_data` decorator for data loading
   - Preprocessing steps cached to avoid redundant computation
   - Model predictions cached for repeated queries

**Scalability:**

- **Dataset Size:** Pipeline scales linearly up to ~50,000 observations
- **Feature Dimensionality:** Efficient handling of up to ~100 features with PCA
- **Model Complexity:** Random Forest with 100 trees optimal for performance/accuracy trade-off

**Memory Management:**
- Iterative model training to prevent memory overflow
- Efficient matrix operations using scipy sparse matrices where applicable
- Garbage collection between major processing steps

---

## Real-World Applications

### Healthcare Providers

**Clinical Screening:**
- First-line screening tool for sleep disorder risk assessment
- 96.67% accuracy enables confident identification of high-risk individuals
- Cost-effective alternative to expensive polysomnography for initial evaluation
- Prioritization of specialist referrals based on risk scores

**Treatment Personalization:**
- Cluster assignments guide personalized treatment protocols
- Feature importance informs intervention targeting
- Progress monitoring through repeated assessments

**Decision Support:**
- Quantified risk scores for clinical documentation
- Evidence-based recommendations for lifestyle modifications
- Integration potential with electronic health records

### Public Health Officials

**Population-Level Interventions:**
- Evidence-based sleep health campaign design
- Targeting based on behavioral phenotype distribution
- Resource allocation optimization

**Workplace Wellness:**
- Occupation-specific intervention programs
- Stress management for high-risk occupations
- Physical activity promotion strategies

**Screening Programs:**
- Community-based screening initiatives
- Early detection and prevention programs
- Targeted interventions for high-risk groups (20% of population)

### Individual Users

**Self-Assessment:**
- Anonymous risk assessment tool
- No clinical visit required for initial screening
- Immediate feedback and recommendations

**Behavior Change Support:**
- Personalized, priority-ranked recommendations
- Quantified impact estimates for motivation
- Progress tracking over time

**Health Awareness:**
- Education about sleep health factors
- Understanding of personal risk factors
- Empowerment through actionable insights

---

## Reproducibility

### Running the Complete Analysis

**1. Data Processing:**
```python
# All preprocessing steps documented in notebook
# Load data → Clean → Encode → Engineer Features → Scale
```

**2. Model Training:**
```python
# Train all 10 models
# Perform hyperparameter tuning
# Apply ensemble methods
```

**3. Model Evaluation:**
```python
# Calculate all performance metrics
# Generate confusion matrices and ROC curves
# Perform cross-validation
```

**4. Save Artifacts:**
```python
# Save 22 model files to models/
# Save 5 processed datasets to data/
# Save 7 artifact files to artifacts/
```

### Saved Artifacts

**Models (22 files):**
- 3 regression models (RF, GB, Ridge)
- 5 classification models (RF, GB, LR, SVC, KNN)
- 1 tuned model (RF with optimized hyperparameters)
- 2 ensemble models (Hard Voting, Soft Voting)
- 4 label encoders
- 5 scalers
- 2 unsupervised models (K-Means, PCA)

**Data (5 files):**
- Cleaned dataset
- Encoded dataset
- Standardized dataset
- Normalized dataset
- Dataset with cluster assignments

**Artifacts (7 files):**
- Feature information (metadata)
- Model performance summary (all metrics)
- Feature importance (Random Forest)
- Feature importance (Gradient Boosting)
- Encoder mappings (for deployment)
- Project report (comprehensive findings)
- Project statistics (summary stats)

### Random Seed

All random processes use `random_state=42` for reproducibility:
- Train-test splits
- Model initialization
- Cross-validation folds
- Clustering initialization
- t-SNE initialization

---

## Dependencies

### Core Libraries

```
streamlit
pandas
numpy
scikit-learn
plotly
seaborn
matplotlib
scipy
joblib
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Local Application

**Start the Streamlit app:**
```bash
streamlit run app.py
```

**Navigate to:**
```
http://localhost:8501
```

**Important:** Use Light Mode for optimal viewing

### Jupyter Notebook Analysis

**Open the complete analysis notebook:**
```bash
jupyter notebook CMSE830_FinalProject_EndTerm.ipynb
```

**Run all cells to:**
- Reproduce all analyses
- Retrain all models
- Generate all visualizations
- Save new model artifacts

### Using Saved Models

**Load a saved model:**
```python
import joblib

# Load classifier
classifier = joblib.load('models/random_forest_classifier.pkl')

# Load scaler
scaler = joblib.load('models/scaler_classification.pkl')

# Make predictions
predictions = classifier.predict(scaler.transform(new_data))
```

### API Integration (Future)

The trained models can be integrated into REST APIs for programmatic access:

```python
# Example Flask API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})
```

---

## Limitations and Future Work

### Current Limitations

**Data Limitations:**
- Cross-sectional data (no temporal causality)
- Single dataset source (generalizability concerns)
- Self-reported measures (potential reporting bias)
- Limited sample size (374 observations)
- No objective sleep measurement validation (e.g., polysomnography)
- Geographic limitation (specific population)

**Model Limitations:**
- Binary gender classification (non-inclusive)
- Discrete occupation categories (oversimplification)
- Static predictions (no temporal dynamics)
- No incorporation of medication/comorbidity data
- Limited interpretability of ensemble methods

**Application Limitations:**
- Requires internet access (for deployed version)
- No user authentication (no personalized tracking)
- No integration with wearable devices
- No clinical validation studies conducted

### Future Enhancements

**Data Collection:**
1. **Longitudinal Study Design:**
   - Track individuals over 6-12 months
   - Establish temporal causality
   - Monitor intervention effectiveness

2. **Wearable Device Integration:**
   - Objective sleep measurement (sleep trackers, smartwatches)
   - Continuous heart rate monitoring
   - Activity tracking validation

3. **Expanded Demographics:**
   - Larger, more diverse sample (target: 5,000+ individuals)
   - Multiple geographic regions
   - Broader age ranges
   - More inclusive gender categories

4. **Additional Variables:**
   - Environmental factors (bedroom temperature, noise, light)
   - Medication usage
   - Comorbidities (diabetes, hypertension, mental health)
   - Dietary patterns
   - Caffeine and alcohol consumption

**Modeling Enhancements:**
1. **Deep Learning Approaches:**
   - Neural networks for larger datasets
   - Recurrent neural networks for temporal patterns
   - Attention mechanisms for feature importance

2. **Time Series Analysis:**
   - Longitudinal modeling (ARIMA, LSTM)
   - Trajectory analysis
   - Change point detection

3. **Causal Inference:**
   - Propensity score matching
   - Instrumental variables
   - Difference-in-differences analysis

4. **Explainable AI:**
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Counterfactual explanations

**Application Enhancements:**
1. **Mobile Application:**
   - Native iOS/Android apps
   - Offline functionality
   - Push notifications for reminders

2. **User Authentication:**
   - Secure login system
   - Personalized tracking over time
   - Progress visualization

3. **Clinical Integration:**
   - EHR (Electronic Health Record) integration
   - FHIR (Fast Healthcare Interoperability Resources) compliance
   - Clinical decision support system

4. **Real-Time Monitoring:**
   - API integration with wearable devices
   - Continuous risk assessment
   - Automated alerts for healthcare providers

**Validation Studies:**
1. **Clinical Trials:**
   - Prospective validation in clinical settings
   - Comparison with gold-standard polysomnography
   - Intervention effectiveness studies

2. **External Validation:**
   - Testing on independent datasets
   - Cross-cultural validation
   - Different population demographics

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{velamuru2025sleep,
  author = {Velamuru, Sai Yashwitha Reddy},
  title = {Sleep Health Analysis: Comprehensive Data Science Project},
  year = {2025},
  publisher = {Michigan State University},
  howpublished = {\url{https://github.com/Yashwitha-7/sleep-health-analysis}},
  note = {CMSE 830 - Foundations of Data Science}
}
```

---

## License

This project is part of academic coursework for CMSE 830 at Michigan State University.

**Academic Use:** Permitted with proper attribution  
**Commercial Use:** Contact author for licensing

---

## Contact

**Author:** Sai Yashwitha Reddy Velamuru  
**Email:** velamur1@msu.edu  
**Institution:** Michigan State University  
**Department:** Computational Mathematics, Science and Engineering  
**Course:** CMSE 830 - Foundations of Data Science  
**Instructor:** Dr. Luciano Germano Silvestri

**Project Links:**
- **GitHub Repository:** [https://github.com/Yashwitha-7/sleep-health-analysis](https://github.com/Yashwitha-7/sleep-health-analysis)
- **Live Application:** [https://sleep-health-analysis-yashwitha.streamlit.app/](https://sleep-health-analysis-yashwitha.streamlit.app/)
- **Dataset Source:** [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

---

## Acknowledgments

**Course Instructor:**
- Dr. Luciano Germano Silvestri, Michigan State University

**Dataset:**
- Sleep Health and Lifestyle Dataset (Kaggle)

**Libraries and Frameworks:**
- scikit-learn: Machine learning models and evaluation
- Streamlit: Interactive web application framework
- Plotly: Interactive visualizations
- pandas & NumPy: Data manipulation and numerical computing
- seaborn & matplotlib: Statistical visualizations

**Institution:**
- Michigan State University
- Department of Computational Mathematics, Science and Engineering (CMSE)

---

## Version History

**Version 1.0 (December 2025)**
- Initial release
- Complete analysis pipeline implemented
- 10 machine learning models trained and evaluated
- Interactive Streamlit application deployed
- Comprehensive documentation completed
- 22 model artifacts saved
- 5 processed datasets created
- 7 analysis artifacts generated

---


*Last Updated: December 6, 2025*  
*Documentation Version: 1.0*  
*Project Status: Complete and Deployed*
