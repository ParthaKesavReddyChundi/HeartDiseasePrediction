# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease in patients using clinical data and logistic regression. This project demonstrates end-to-end ML workflow including data preprocessing, exploratory data analysis, model training, and evaluation.

## 🎯 Project Overview

This project aims to build a predictive model that can identify patients at risk of heart disease based on 13 medical features. Using the UCI Heart Disease dataset and Logistic Regression, we achieve robust classification performance with interpretable results.

**Dataset**: 920 patient records with 13 clinical features
**Model**: Logistic Regression
**Target**: Binary classification (Disease / No Disease)

## 📊 Dataset

### Source
UCI Heart Disease Dataset (publicly available)

### Key Statistics
- **Total Samples**: 920 patients
- **Features**: 13 medical attributes
- **Class Distribution**: 55.3% with disease, 44.7% without disease
- **Missing Data**: Handled via median imputation (numerical) and mode imputation (categorical)

### Features Used
| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Age in years |
| `sex` | Binary | Male/Female |
| `trestbps` | Numerical | Resting blood pressure (mmHg) |
| `chol` | Numerical | Serum cholesterol (mg/dl) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl |
| `thalch` | Numerical | Maximum heart rate achieved |
| `exang` | Binary | Exercise induced angina |
| `oldpeak` | Numerical | ST depression by exercise |
| `cp` | Categorical | Chest pain type (3 categories) |
| `restecg` | Categorical | Resting ECG results (2 categories) |

**Features Dropped** (due to >30% missing data):
- `ca` (coronary artery calcification) - 66% missing
- `thal` (thalassemia) - 52% missing
- `slope` (ST slope) - 33% missing

## 🔧 Data Preprocessing

1. **Missing Value Imputation**
   - Numerical features: Median imputation (robust to outliers)
   - Categorical features: Mode imputation (most frequent value)

2. **Feature Encoding**
   - Binary features: Direct mapping to 0/1
   - Categorical features: One-Hot Encoding with `drop_first=True`

3. **Feature Scaling**
   - StandardScaler applied to numerical features (mean=0, std=1)
   - Essential for logistic regression convergence

## 📈 Model Performance

### Evaluation Metrics (Test Set - 184 samples)
- **Accuracy**: 89.13%
- **Precision**: 0.8919
- **Recall**: 0.8627
- **F1 Score**: 0.8770
- **ROC-AUC**: 0.9443

### Confusion Matrix
```
                  Predicted Negative  Predicted Positive
Actual Negative             73                9
Actual Positive             15               87
```

- **True Negatives** (TN): 73 - Correctly identified healthy patients
- **True Positives** (TP): 87 - Correctly identified disease patients
- **False Negatives** (FN): 15 - Missed disease cases ⚠️
- **False Positives** (FP): 9 - False alarms

## 🔍 Feature Importance

Features ranked by impact on disease prediction (Logistic Regression Coefficients):

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|------------|-----------|-----------------|
| Male Sex | +1.166 | 3.21 | Males 3.2x more likely to have disease |
| Exercise Induced Angina | +1.013 | 2.75 | Increases disease odds by 175% |
| ST Depression (oldpeak) | +0.517 | 1.68 | Higher depression more likely disease |
| Atypical Angina | -1.852 | 0.16 | Decreases disease odds by 84% |
| Non-Anginal Pain | -1.100 | 0.33 | Decreases disease odds by 67% |
| Max Heart Rate | -0.336 | 0.71 | Higher heart rate slightly protective |

## 🚀 Quick Start

### Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HeartDiseasePrediction.git
cd HeartDiseasePrediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook HeartDiseasePrediction.ipynb
   ```

2. **Five Main Sections**
   - **Step 1 - EDA**: Dataset exploration with 6 visualizations
   - **Step 2 - Preprocessing**: Data cleaning and feature engineering
   - **Step 3 - Model Training**: Train logistic regression (80/20 split)
   - **Step 4 - Evaluation**: Comprehensive metrics and ROC curve
   - **Step 5 - Interpretation**: Feature importance and real-world predictions

3. **Make Predictions on New Patients**
   ```python
   # Example: Predict disease risk for a new patient
   new_patient = pd.DataFrame({
       'age': [62],
       'sex': [1],  # Male
       'trestbps': [150],
       'chol': [280],
       # ... other features
   })
   prediction = model.predict_proba(new_patient)
   ```

## 📁 Project Structure

```
HeartDiseasePrediction/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── HeartDiseasePrediction.ipynb     # Main notebook (5 steps)
└── heart_disease_uci.csv            # UCI dataset (920 samples)
```

## 📊 Visualizations Included

- **EDA**: Target distribution, age histogram, sex vs disease, chest pain analysis
- **Model Performance**: Confusion matrix heatmap, ROC curve, metrics comparison
- **Feature Importance**: Coefficient plot, odds ratio visualization

## 🔑 Key Insights

1. **Male patients have 3.2x higher odds** of heart disease
2. **Exercise-induced angina** is one of the strongest predictors
3. **Atypical chest pain** suggests lower disease risk
4. **Maximum heart rate** shows inverse relationship with disease
5. **Model achieves 89% accuracy** with strong ROC-AUC (0.944)

## ⚠️ Clinical Disclaimer

This model is **educational only** and should **NOT** be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 🔬 Model Evaluation Approach

- **Train/Test Split**: 80% training, 20% testing with stratification
- **Stratified Split**: Maintains class distribution (55/45) in both sets
- **Random Seed**: `random_state=42` for reproducibility

## 📚 Data Sources

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) - Heart Disease datasets from multiple institutions

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Try other models (Random Forest, SVM, Neural Networks)
- Implement cross-validation
- Add feature selection techniques
- Handle imbalanced data with SMOTE
- Build a web interface for predictions

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Created as a machine learning demonstration project.

## 📧 Questions & Support

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

---

**Last Updated**: March 2026
**Model Status**: Production-ready for educational purposes
**Test Accuracy**: 89.13%
