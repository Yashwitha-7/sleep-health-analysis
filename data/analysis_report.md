
# Sleep Health and Lifestyle Analysis Report

## Executive Summary
- **Dataset Size**: 374 records, 16 features
- **Analysis Date**: 2025-10-15
- **Sleep Disorder Prevalence**: 41.4%

## Key Findings

### 1. Sleep Quality Factors
- Average sleep duration: 7.13 hours
- Average sleep quality score: 7.31/10
- 155 individuals (41.4%) sleep less than 7 hours

### 2. Lifestyle Impact
- Strong correlation between physical activity and sleep quality: 0.193
- Negative correlation between stress and sleep quality: -0.899

### 3. Demographics
- Gender distribution: {'Male': 189, 'Female': 185}
- Age range: 27-59 years
- Most common occupation: Nurse

### 4. Health Indicators
- Average heart rate: 70.2 BPM
- Average blood pressure: 128.6/84.6 mmHg

## Data Processing Summary

### Transformations Applied
1. **Missing Value Handling**: Sleep Disorder NaN values converted to 'None'
2. **Feature Engineering**: Created Sleep Efficiency, Activity-Stress Ratio, Sleep Deficit
3. **Encoding**: Label encoding for categorical variables
4. **Scaling**: Both standardization and min-max normalization applied
5. **Binning**: Created age groups, sleep categories, stress categories

### Datasets Generated
- sleep_health_cleaned.csv: Main dataset with engineered features
- sleep_health_ml_ready.csv: Fully encoded for machine learning
- sleep_health_standardized.csv: Standardized features (mean=0, std=1)
- sleep_health_normalized.csv: Normalized features (range 0-1)

## Next Steps
1. Deploy interactive Streamlit dashboard
2. Implement predictive models for sleep disorder classification
3. Create personalized sleep quality recommendations
4. Develop risk assessment tool

---
*Report generated from comprehensive EDA and data preprocessing*
