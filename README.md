# 🔒 Fraud Detection Using Decision Trees

A comprehensive machine learning project that demonstrates transaction fraud classification using interpretable Decision Tree models. This project showcases the complete ML workflow from data generation to model deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Understanding the Features](#understanding-the-features)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project builds a machine learning model to classify financial transactions as **fraudulent** or **legitimate** using **real-world credit card fraud data**. The project uses the popular [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains 284,807 transactions with only 492 frauds (0.172% fraud rate) - a highly imbalanced real-world scenario.

Decision Trees are used for their interpretability, making it easy to understand which factors contribute to fraud detection and explain decisions to stakeholders.

### Why Decision Trees?

- **Interpretable**: Easy to visualize and understand the decision-making process
- **No feature scaling required**: Works with raw features (though we scale for better performance)
- **Handles mixed data types**: Categorical and numerical features
- **Fast predictions**: Efficient for real-time fraud detection
- **Feature importance**: Identifies key fraud indicators
- **Explainable AI**: Can trace exact decision path for any prediction

## ✨ Features

### Dataset Support
- **Real Kaggle Dataset**: Full integration with the Credit Card Fraud Detection dataset
- **Sample Data**: Mimics real dataset structure when full data unavailable
- **Synthetic Data**: Generate custom fraud patterns for testing

### Machine Learning Pipeline
- **Data Preprocessing**: Handles PCA features, scaling, and imbalanced classes
- **Model Training**: Decision Tree with optimized hyperparameters
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Imbalanced Data Handling**: Class weights and appropriate metrics

### Visualizations
- **Confusion Matrix**: Both counts and percentages
- **ROC Curve**: Model discrimination ability
- **Precision-Recall Curve**: Performance on imbalanced data
- **Feature Importance**: Key fraud indicators
- **Decision Tree**: Visual representation of decision logic

### Production Ready
- **Real-time Prediction**: Classify new transactions instantly
- **Modular Architecture**: Easy to extend and customize
- **Error Handling**: Graceful fallbacks and informative messages
- **Reproducible Results**: Fixed random seeds

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection-decision-trees.git
cd fraud-detection-decision-trees
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Download the Kaggle dataset** (recommended):
   - Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv` (143 MB)
   - Place it in the project root directory

   **Note**: You can still run the project without downloading the dataset - it will automatically use sample data that mimics the real structure.

## 💻 Usage

### Quick Start

**With Real Dataset:**
```bash
# Make sure creditcard.csv is in the project directory
python fraud_detection.py
```

**Without Real Dataset (uses sample data):**
```bash
# The script automatically falls back to sample data
python fraud_detection.py
```

This will:
1. Load the real Kaggle dataset (or create sample data if unavailable)
2. Preprocess and split data into training (80%) and testing (20%)
3. Train a Decision Tree classifier with optimized parameters
4. Evaluate performance with multiple metrics
5. Generate 5 visualization plots
6. Display comprehensive results

### Understanding the Dataset

**Real Kaggle Dataset Features:**
- **V1-V28**: PCA-transformed features (anonymized for privacy)
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction amount
- **Class**: 0 = Legitimate, 1 = Fraud

The dataset is highly imbalanced (0.172% fraud), making it a realistic challenge.

### Custom Configuration

Modify the main() function for custom behavior:

```python
# Use real dataset
df = fraud_detector.load_real_dataset('creditcard.csv')

# Use sample data (mimics real structure)
df = fraud_detector.load_sample_real_data(n_samples=50000)

# Use synthetic data (custom features)
df = fraud_detector.generate_synthetic_data(n_samples=10000, fraud_ratio=0.1)

# Adjust train/test split
fraud_detector.prepare_data(df, test_size=0.3)

# Train with custom hyperparameters
fraud_detector.train_model(max_depth=20, min_samples_split=200, min_samples_leaf=100)

# Enable hyperparameter optimization (takes longer but better results)
best_params = fraud_detector.optimize_hyperparameters(cv=5)
```

### Predicting New Transactions

**For Real Dataset (V1-V28 features):**
```python
# Must match the exact feature names from training
new_transaction = {
    'V1': -1.359807,
    'V2': -0.072781,
    # ... V3 through V27 ...
    'V28': -0.021053,
    'Amount_scaled': 0.244964,
    'Time_scaled': 0.000000
}

prediction, probability = fraud_detector.predict_new_transaction(new_transaction)
```

**For Synthetic Dataset:**
```python
new_transaction = {
    'amount': 250,
    'time_of_day': 2,
    'distance_from_home': 150,
    'distance_from_last': 100,
    'ratio_to_median': 4.5,
    'repeat_retailer': 0,
    'used_chip': 0,
    'used_pin': 0,
    'online_order': 1
}

prediction, probability = fraud_detector.predict_new_transaction(new_transaction)
```

## 📁 Project Structure

```
fraud-detection-decision-trees/
│
├── fraud_detection.py          # Main script with complete pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
│
├── creditcard.csv              # Real Kaggle dataset (download separately)
│
├── outputs/                     # Generated visualizations (created at runtime)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── decision_tree.png
│
├── notebooks/                   # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
│
└── data/                        # Additional datasets (optional)
    ├── sample_data.csv
    └── synthetic_data.csv
```

## 📊 Model Performance

Performance on the **real Kaggle dataset** (highly imbalanced - 0.172% fraud):

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | ~99.9% | Overall correct predictions |
| **Precision** | ~75-85% | Of flagged frauds, how many are real |
| **Recall** | ~75-85% | Of real frauds, how many we catch |
| **F1 Score** | ~75-85% | Balanced precision-recall metric |
| **ROC-AUC** | ~85-90% | Overall discrimination ability |
| **PR-AUC** | ~70-80% | Performance on imbalanced data |

**Important Notes:**
- Accuracy is misleading for imbalanced data (even predicting all legitimate gives 99.8% accuracy)
- Precision, Recall, F1, and PR-AUC are more meaningful metrics
- Trade-off between catching fraud (recall) and false alarms (precision)
- Performance varies with hyperparameter tuning

Performance on **synthetic dataset** (balanced - 10% fraud):

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97% |
| **Precision** | ~95% |
| **Recall** | ~92% |
| **F1 Score** | ~93% |
| **ROC-AUC** | ~98% |

*These are approximate values - actual results vary based on random seed and hyperparameters*

### What These Metrics Mean for Fraud Detection

- **Precision**: If we flag a transaction as fraud, what's the probability it's actually fraud? (Minimize false alarms)
- **Recall (Sensitivity)**: Of all actual fraud cases, what percentage do we catch? (Minimize missed frauds)
- **F1 Score**: Harmonic mean of precision and recall (balanced metric when both matter)
- **ROC-AUC**: Overall ability to distinguish fraud from legitimate across all thresholds
- **PR-AUC**: Precision-Recall AUC - better metric for highly imbalanced datasets

**Business Trade-offs:**
- **High Precision**: Fewer false alarms, but might miss some fraud
- **High Recall**: Catch more fraud, but more false alarms (legitimate transactions flagged)
- In fraud detection, typically recall is prioritized (better to investigate more than miss fraud)

## 📈 Visualizations

The project generates five key visualizations:

### 1. Confusion Matrix (Counts & Percentages)
Shows the distribution of correct and incorrect predictions in both absolute numbers and percentages.

### 2. ROC Curve
Illustrates the trade-off between true positive rate and false positive rate across different classification thresholds.

### 3. Precision-Recall Curve
Especially important for imbalanced datasets - shows precision-recall trade-off. More informative than ROC for fraud detection.

### 4. Feature Importance
Ranks features by their contribution to fraud detection. For the Kaggle dataset, shows which PCA components are most predictive.

### 5. Decision Tree Visualization
Visual representation of the decision-making process (limited to 3 levels for readability). Shows the actual logic the model uses.

## 🔍 Understanding the Features

### Real Kaggle Dataset
Due to privacy, most features are PCA-transformed:

| Feature | Description |
|---------|-------------|
| **V1-V28** | PCA components (anonymized financial features) |
| **Time** | Seconds elapsed since first transaction in dataset |
| **Amount** | Transaction amount (scaled in our preprocessing) |
| **Class** | 0 = Legitimate, 1 = Fraud |

**Note**: While we can't interpret individual V1-V28 features directly, feature importance tells us which components best discriminate fraud.

### Synthetic Dataset Features
When using synthetic data, features have clear interpretations:

| Feature | Description | Fraud Pattern |
|---------|-------------|---------------|
| **amount** | Transaction amount ($) | Fraudulent transactions tend to be larger |
| **time_of_day** | Hour of transaction (0-23) | Fraud more common at unusual hours (2-4 AM, 10-11 PM) |
| **distance_from_home** | Distance from home address (km) | Fraud often occurs far from home |
| **distance_from_last** | Distance from previous transaction (km) | Large jumps in location indicate fraud |
| **ratio_to_median** | Ratio to user's median spending | Fraud transactions often much higher than normal |
| **repeat_retailer** | Previously used retailer (0/1) | Fraud more likely at new retailers |
| **used_chip** | Chip card used (0/1) | Fraud less likely with chip |
| **used_pin** | PIN entered (0/1) | Fraud less likely with PIN |
| **online_order** | Online transaction (0/1) | Fraud slightly more common online |

## 🔮 Future Improvements

Potential enhancements for the project:

### Data & Features
- [ ] **Feature Engineering**: Create interaction terms from PCA components
- [ ] **Temporal Features**: Extract hour/day patterns from Time feature
- [ ] **External Data**: Integrate with other fraud datasets
- [ ] **Anomaly Detection**: Add unsupervised methods (Isolation Forest, Autoencoder)

### Models & Algorithms
- [ ] **Ensemble Methods**: Compare with Random Forests, XGBoost, LightGBM
- [ ] **Deep Learning**: Implement neural network approaches (LSTM for sequences)
- [ ] **SMOTE/ADASYN**: Advanced oversampling for imbalanced data
- [ ] **Cost-sensitive Learning**: Assign different costs to false positives vs false negatives
- [ ] **Model Stacking**: Combine multiple models for better performance

### Explainability & Interpretation
- [ ] **SHAP Values**: Individual prediction explanations
- [ ] **LIME**: Local interpretable model-agnostic explanations
- [ ] **Counterfactual Explanations**: "What would make this legitimate?"
- [ ] **Rule Extraction**: Convert tree to human-readable rules

### Deployment & Production
- [ ] **REST API**: Flask/FastAPI for real-time predictions
- [ ] **Streamlit Dashboard**: Interactive web interface
- [ ] **Docker Container**: Containerize the application
- [ ] **Model Monitoring**: Track drift and performance degradation
- [ ] **A/B Testing Framework**: Compare models in production
- [ ] **Batch Prediction Pipeline**: Process large transaction batches

### Performance & Optimization
- [ ] **Threshold Optimization**: Find optimal probability threshold for business needs
- [ ] **Calibration**: Calibrate probability outputs (Platt scaling, isotonic regression)
- [ ] **Feature Selection**: Automated feature selection methods
- [ ] **AutoML**: Automated hyperparameter tuning with Optuna/Ray Tune

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Your Name - [@yourhandle](https://twitter.com/KiprutoLagatIK)

Project Link: [https://github.com/isaacLagat/fraud-detection-decision-trees](https://github.com/isaacLagat/fraud-detection-decision-trees)

## 🙏 Acknowledgments

- Scikit-learn documentation and community
- Various fraud detection research papers and tutorials
- Open-source ML community

## 📚 Resources & References

### Dataset
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original research paper: [Calibrating Probability with Undersampling for Unbalanced Classification](https://www.researchgate.net/publication/283349138_Calibrating_Probability_with_Undersampling_for_Unbalanced_Classification)

### Documentation & Tutorials
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Understanding Decision Trees](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/stable/)
- [Precision-Recall vs ROC Curves](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

### Research Papers
- Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32
- Dal Pozzolo, A. et al. (2015). "Learned lessons in credit card fraud detection from a practitioner perspective"

### Related Projects
- [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- [Awesome Fraud Detection](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)

---

⭐ If you found this project helpful, please consider giving it a star!
