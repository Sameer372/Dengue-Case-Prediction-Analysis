

\# Dengue Case Prediction and Analysis



An experimental machine learning project for dengue case analysis, time-series forecasting, spatial exploration, and blood reserve planning using historical dengue data from Bangladesh.



\## Project Overview



This project explores historical dengue cases using machine learning, deep learning, statistical forecasting, and ensemble learning techniques.



It also includes regional spatial analysis, dengue risk classification, and an experimental blood reserve planning framework.



\## Project Objectives



\- Analyze historical dengue trends.

\- Forecast future dengue cases.

\- Compare different machine learning models.

\- Explore regional dengue case distributions.

\- Perform experimental spatial analysis.

\- Estimate potential blood reserve requirements.



\## Models and Techniques



\- LSTM

\- GRU

\- SimpleRNN

\- Prophet

\- XGBoost Stacking Ensemble

\- DBSCAN Clustering

\- Moran's I



\## Dataset



\- File: Dengue.xlsx

\- Records: 51

\- Features: 20

\- Coverage: September 2019 – October 2025

\- Geographic Coverage: Eight divisions of Bangladesh.



Dataset source and usage permission should be documented before public release.



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- TensorFlow / Keras

\- Prophet

\- XGBoost

\- Jupyter Notebook



\## Project Structure



```text

Dengue-Case-Prediction-Analysis/

├── Dengue.xlsx

├── dengue\_analysis.ipynb

├── dengue\_analysis.py

├── model\_flowchart.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the Python script:



```bash

python dengue\_analysis.py

```



Alternatively, open the Jupyter Notebook:



```bash

jupyter notebook dengue\_analysis.ipynb

```



\## Limitations



\- The dataset contains missing calendar periods.

\- Forecasting results require further validation.

\- The notebook and Python script have implementation differences.

\- Spatial analysis and blood reserve estimates are experimental.



\## Future Improvements



\- Improve missing-period handling.

\- Synchronize notebook and Python implementations.

\- Improve forecasting validation.

\- Refine spatial analysis.

\- Validate experimental assumptions using reliable external data.



