
# Dengue Case Prediction and Analysis

An experimental machine learning project focused on dengue case analysis, time-series forecasting, spatial exploration, and blood reserve planning using historical dengue data from Bangladesh.

## Project Overview

This project explores historical dengue cases in Bangladesh using machine learning, deep learning, statistical forecasting, and ensemble learning techniques.

It combines multiple forecasting models to investigate dengue trends and estimate future case patterns. The project also includes regional spatial analysis, dengue risk classification, and an experimental blood reserve planning framework.

## Project Objectives

- Analyze historical dengue case trends in Bangladesh.
- Forecast future dengue cases using machine learning and deep learning models.
- Compare different forecasting approaches.
- Explore regional dengue case distributions across eight divisions.
- Investigate spatial patterns using clustering and spatial autocorrelation techniques.
- Perform experimental dengue risk classification.
- Estimate potential blood reserve requirements through an experimental planning framework.

## Dataset

The project uses historical dengue data from Bangladesh.

**Dataset Information:**

- File Name: `Dengue.xlsx`
- Total Records: 51
- Total Features: 20
- Data Coverage: September 2019 – October 2025
- Geographic Coverage: Eight divisions of Bangladesh.

The dataset includes monthly dengue cases, deaths, and division-wise case and death records.

### Dataset Source and Usage Permission

- Dataset: Dengue.xlsx
- Public Distribution Permission: Confirmed.
- Original Source: [Add the original dataset source URL here.]

The dataset is included in this repository with permission for public distribution.

## Methodology

The project follows several analytical and modeling approaches.

### 1. Data Preprocessing

- Data loading and inspection.
- Data cleaning and preparation.
- Date generation using year and month information.
- Chronological sorting.
- Feature scaling.
- Time-series sequence preparation.

### 2. Time-Series Forecasting

The following models are explored for dengue case forecasting:

- **LSTM (Long Short-Term Memory):** A recurrent neural network architecture for learning sequential patterns.
- **GRU (Gated Recurrent Unit):** A recurrent architecture for sequence modeling.
- **SimpleRNN:** A basic recurrent neural network for time-series analysis.
- **Prophet:** A statistical forecasting model for analyzing temporal trends.

### 3. Stacked Ensemble Learning

An XGBoost Regressor is used as the meta-learner in a stacking ensemble approach to combine predictions from the base forecasting models.

### 4. Spatial Analysis

The project explores regional spatial patterns using:

- **DBSCAN:** Density-based clustering for geographic exploration.
- **Moran's I:** Spatial autocorrelation analysis.

### 5. Dengue Risk Classification

An experimental classification approach is included to explore dengue risk categories.

### 6. Future Forecasting

The forecasting pipeline generates experimental dengue case predictions for the next six months based on historical patterns.

### 7. Blood Reserve Planning

An experimental framework estimates potential blood reserve requirements and regional allocation using predefined assumptions, including severity rates, estimated units per patient, safety buffers, and hotspot adjustments.

## Model Evaluation

The project explores different evaluation metrics for forecasting and classification tasks.

**Forecasting Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Weighted Mean Absolute Percentage Error (WMAPE)

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

Actual performance values should be interpreted alongside the dataset limitations and validation methodology.

## Technologies Used

**Programming Language**
- Python

**Data Analysis and Visualization**
- Pandas
- NumPy
- Matplotlib
- Seaborn

**Machine Learning and Deep Learning**
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Prophet

**Development Environment**
- Jupyter Notebook

## Project Structure

```text
Dengue-Case-Prediction-Analysis/
│
├── Dengue.xlsx
├── dengue_analysis.ipynb
├── dengue_analysis.py
├── model_flowchart.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sameer372/Dengue-Case-Prediction-Analysis.git
```

### 2. Navigate to the Project Directory

```bash
cd Dengue-Case-Prediction-Analysis
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Run the Python Pipeline

```bash
python dengue_analysis.py
```

### Run the Jupyter Notebook

```bash
jupyter notebook dengue_analysis.ipynb
```

Alternatively, open the notebook using JupyterLab or Visual Studio Code.

### Generate the Model Flowchart

```bash
python model_flowchart.py
```

## Limitations

- The dataset contains missing calendar periods, which may affect time-series sequence preparation and forecasting.
- The relatively small dataset limits the amount of historical information available for model training.
- The Jupyter Notebook and Python script currently contain implementation differences.
- Forecasting results require further validation using appropriate chronological evaluation methods.
- Spatial clustering results are exploratory and should not be interpreted as confirmed dengue hotspots.
- Blood reserve estimates rely on predefined experimental assumptions rather than validated medical guidelines.

## Future Improvements

- Improve missing-period handling and data preprocessing.
- Synchronize the Jupyter Notebook and Python implementations.
- Improve time-series validation and forecasting reliability.
- Explore additional forecasting and ensemble learning techniques.
- Refine regional spatial analysis and hotspot identification.
- Validate dengue risk classification approaches.
- Improve blood reserve estimation using reliable epidemiological data and expert-reviewed assumptions.

## Disclaimer

This project is developed for academic and research purposes.

The forecasting outputs, spatial analysis, risk classification, and blood reserve estimates are experimental. They should not be considered medical advice or used as standalone evidence for clinical, public health, or blood bank operational decisions.

## Author

**Shaikh Sameer**

GitHub: [@Sameer372](https://github.com/Sameer372)

---

If you find this project useful, feel free to explore the repository and its implementation.
