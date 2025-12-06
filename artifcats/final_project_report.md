
# SLEEP HEALTH ANALYSIS - COMPREHENSIVE PROJECT REPORT

**Student:** Yashwitha Velamuru  
**Course:** CMSE 830 - Foundations of Data Science  
**Institution:** Michigan State University  
**Date:** December 03, 2025

---

## EXECUTIVE SUMMARY

This project presents a comprehensive analysis of sleep health data, implementing advanced
data science techniques to predict sleep quality and identify individuals at risk for sleep
disorders. The analysis incorporates exploratory data analysis, feature engineering, multiple
machine learning models, hyperparameter tuning, ensemble methods, and unsupervised learning.

### Key Achievements:
- ✓ Processed and analyzed 374 observations with 26 features
- ✓ Engineered 13 new features for enhanced predictive power
- ✓ Trained 9 machine learning models
- ✓ Achieved 100.00% variance explanation for sleep quality prediction (R²=1.0000)
- ✓ Achieved 96.00% accuracy for sleep disorder classification
- ✓ Identified 9 distinct sleep health profiles through clustering

---

## 1. DATA COLLECTION AND PREPROCESSING

### Dataset Information
- **Source:** Sleep Health and Lifestyle Dataset (Kaggle)
- **Observations:** 374
- **Original Features:** 13
- **Final Features:** 26

### Data Quality Assessment
- **Duplicates:** 0 rows (removed if present)
- **Missing Values:** Properly handled (Sleep Disorder NaN treated as "None" category)
- **Outliers:** Identified using IQR method, retained as clinically meaningful
- **Data Validation:** All physiological ranges verified

### Data Processing Implemented
1. **Encoding:** 4 categorical variables encoded using LabelEncoder
2. **Feature Engineering:** 13 new features created
3. **Scaling:** 3 methods (StandardScaler, MinMaxScaler, RobustScaler)
4. **Imputation:** Multiple techniques demonstrated (Mean, KNN, Iterative)

---

## 2. EXPLORATORY DATA ANALYSIS

### Sleep Quality Insights
- **Mean Quality Score:** 7.31/10
- **Mean Sleep Duration:** 7.13 hours
- **Average Sleep Deficit:** 0.87 hours

### Sleep Disorder Prevalence
- **None:** 58.56%
- **Sleep Apnea:** 20.86%
- **Insomnia:** 20.59%

### Key Correlations with Sleep Quality
- **Sleep Duration:** +0.883
- **Stress Level:** -0.899
- **Physical Activity Level:** +0.193
- **Heart Rate:** -0.660
- **Sleep_Efficiency:** +0.978

---

## 3. MODEL DEVELOPMENT - REGRESSION

### Objective
Predict sleep quality score (1-10 scale) using lifestyle and physiological features.

### Models Trained
1. **Gradient Boosting**
   - R² Score: 1.0000
   - RMSE: 0.0000
   - MAE: 0.0000
2. **Random Forest**
   - R² Score: 0.9997
   - RMSE: 0.0221
   - MAE: 0.0039
3. **Ridge Regression**
   - R² Score: 0.9948
   - RMSE: 0.0886
   - MAE: 0.0548
4. **Support Vector Regressor**
   - R² Score: 0.9726
   - RMSE: 0.2035
   - MAE: 0.0932

### Best Performing Model
**Gradient Boosting**
- **R² Score:** 1.0000 (100.00% variance explained)
- **RMSE:** 0.0000
- **MAE:** 0.0000
- **Interpretation:** The model explains 100.00% of the variance in sleep quality scores

---

## 4. MODEL DEVELOPMENT - CLASSIFICATION

### Objective
Classify individuals into three sleep disorder categories: None, Insomnia, Sleep Apnea.

### Models Trained
1. **Random Forest**
   - Accuracy: 0.9600 (96.00%)
   - F1-Score: 0.9606
2. **Gradient Boosting**
   - Accuracy: 0.9600 (96.00%)
   - F1-Score: 0.9606
3. **Support Vector Classifier**
   - Accuracy: 0.9600 (96.00%)
   - F1-Score: 0.9600
4. **Logistic Regression**
   - Accuracy: 0.9467 (94.67%)
   - F1-Score: 0.9471
5. **K-Nearest Neighbors**
   - Accuracy: 0.9467 (94.67%)
   - F1-Score: 0.9472

### Best Performing Model
**Random Forest**
- **Accuracy:** 0.9600 (96.00%)
- **Precision:** 0.9617
- **Recall:** 0.9600
- **F1-Score:** 0.9606

---

## 5. ADVANCED TECHNIQUES

### Hyperparameter Tuning
- **Method:** GridSearchCV with 5-fold cross-validation
- **Configurations Tested:** 216
- **Best CV Score:** 0.9030
- **Test Set Improvement:** 0.0133 (+1.39%)

### Ensemble Methods
- **Hard Voting Accuracy:** 0.9600
- **Soft Voting Accuracy:** 0.9600
- **Best Ensemble:** Hard Voting
- **Performance:** 0.9600

### Cross-Validation
- **Method:** 10-fold stratified cross-validation
- **Best Model:** SVC
- **Best CV Score:** 0.8723 ± 0.1234

---

## 6. FEATURE IMPORTANCE ANALYSIS

### Top 10 Most Important Features
1. **Diastolic_BP:** 0.1302
2. **Systolic_BP:** 0.1218
3. **Occupation_Encoded:** 0.1214
4. **BMI_Category_Encoded:** 0.1197
5. **Age:** 0.1043
6. **Sleep_Efficiency:** 0.0715
7. **Sleep Duration:** 0.0699
8. **Sleep_Deficit:** 0.0690
9. **Activity_Stress_Ratio:** 0.0463
10. **Heart Rate:** 0.0377

### Key Insights
- **Primary Driver:** Diastolic_BP (importance: 0.1302)
- **Secondary Factor:** Systolic_BP (importance: 0.1218)
- **Cumulative Explanation:** Top 5 features explain 59.74% of variance

---

## 7. CLUSTERING ANALYSIS

### Methodology
- **Algorithm:** K-Means Clustering
- **Optimal Clusters:** 9 (determined via Elbow Method and Silhouette Score)
- **Best Silhouette Score:** 0.6292

### Validation
- **Chi-Square Test:** χ² = 506.9569, p = 0.0000
- **Significance:** Yes - Clusters significantly associated with sleep disorders

### Dimensionality Reduction
- **PCA:** 2 components explaining 64.73% variance
- **t-SNE:** Non-linear reduction revealing distinct groupings

---

## 8. KEY FINDINGS AND INSIGHTS

### For Healthcare Providers
1. **Stress Management Priority:** Stress shows strongest negative correlation with sleep quality (r=-0.899)
2. **Physical Activity Threshold:** Individuals with ≥60 min/day activity show better sleep outcomes
3. **High-Risk Identification:** 155 individuals with sleep deficit >1 hour
4. **Occupational Interventions:** Target professions with elevated disorder rates

### For Public Health Policy
1. **Sleep Apnea Prevalence:** 20.9% - screening programs recommended
2. **Insomnia Prevalence:** 20.6% - stress management programs needed
3. **Population Sleep Deficit:** 81.0% not meeting 8-hour recommendation
4. **Preventive Measures:** Focus on modifiable factors (stress, activity, lifestyle)

### For Individuals
1. **Optimal Sleep Duration:** 7-8 hours (current average: 7.1h)
2. **Stress Management:** Keep stress levels below 5/10
3. **Activity Goals:** Maintain 60+ minutes of physical activity daily
4. **Health Monitoring:** Track resting heart rate (ideal <70 bpm)

---

## 9. MODEL DEPLOYMENT RECOMMENDATIONS

### Production Models
1. **Sleep Quality Prediction:** Gradient Boosting (R²=1.0000)
2. **Disorder Classification:** Random Forest (Accuracy=0.9600)
3. **Ensemble Option:** Hard Voting (Accuracy=0.9600)

### Implementation Requirements
- All models saved in `/models` directory
- Encoders and scalers ready for deployment
- Feature engineering pipeline documented
- Performance baselines established

---

## 10. LIMITATIONS AND FUTURE WORK

### Current Limitations
- Single dataset source (though comprehensive processing)
- Cross-sectional data (no temporal analysis)
- Self-reported measures for some features
- Limited demographic diversity information

### Future Enhancements
1. **Data Collection:** Incorporate additional external datasets
2. **Temporal Analysis:** Time series analysis of sleep patterns
3. **Deep Learning:** Neural networks for complex pattern recognition
4. **Real-time Monitoring:** Integration with wearable device data
5. **Personalized Recommendations:** Individual-level intervention strategies

---

## 11. TECHNICAL SPECIFICATIONS

### Software and Libraries
- Python 3.9+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Jupyter Notebook/Lab

### Computational Requirements
- Standard laptop/desktop (no GPU required)
- Processing time: ~10-15 minutes for complete analysis
- Memory: <2GB RAM

### Files Generated
- **Models:** 17 trained models (.pkl files)
- **Data:** 5 processed datasets (.csv files)
- **Artifacts:** Performance summaries, feature importance (.json, .csv)
- **Reports:** Comprehensive documentation

---

## 12. CONCLUSIONS

This project demonstrates a comprehensive approach to sleep health analysis, combining:
- ✓ Rigorous data preprocessing and quality assessment
- ✓ Advanced feature engineering and transformation
- ✓ Multiple machine learning approaches (supervised and unsupervised)
- ✓ Robust model validation and ensemble methods
- ✓ Clear, actionable insights for stakeholders

The developed models achieve strong predictive performance (100.00% variance
explained for quality, 96.00% accuracy for disorders) and provide a solid
foundation for practical sleep health interventions and decision support systems.

---

## ACKNOWLEDGMENTS

- **Data Source:** Sleep Health and Lifestyle Dataset (Kaggle)
- **Course:** CMSE 830 - Foundations of Data Science
- **Institution:** Michigan State University
- **Instructor:** Dr. Luciano G. Silvestri

---

**Report Generated:** December 03, 2025 at 08:48 PM
**Project Status:** Complete ✓
**Ready for Deployment:** Yes ✓
