import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from prophet import Prophet
import xgboost as xgb
import warnings
import sys

warnings.filterwarnings('ignore')

# Redirect stdout to both console and a file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

f = open('summary.txt', 'w')
sys.stdout = Tee(sys.stdout, f)

# Load Data
file_path = 'Dengue.xlsx'
df = pd.read_excel(file_path)

# Display first few rows
print(df.head())
print(df.columns)

# Data Cleaning
# Create Date column from Year and Month
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].map(month_map)
df['Date'] = pd.to_datetime(df[['Year', 'Month_Num']].assign(DAY=1).rename(columns={'Year': 'year', 'Month_Num': 'month', 'DAY': 'day'}))

# Sort by Date
df = df.sort_values('Date').reset_index(drop=True)

# Handle missing values (Forward Fill)
df = df.ffill()

# Select Target Column
target_col = 'Affected' # Total Cases
data = df[['Date', target_col]].copy()
data.columns = ['Date', 'Dengue_Cases']

print(data.head())

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Dengue_Cases'])
plt.title('Dengue Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.savefig('dengue_cases_over_time.png')
# plt.show()

# Data Splitting
train_size = int(len(data) * 0.70)
val_size = int(len(data) * 0.15)
test_size = len(data) - train_size - val_size

train_data = data.iloc[:train_size].copy()
val_data = data.iloc[train_size:train_size+val_size].copy()
test_data = data.iloc[train_size+val_size:].copy()

print(f"Train Size: {len(train_data)}")
print(f"Validation Size: {len(val_data)}")
print(f"Test Size: {len(test_data)}")
print(f"Test Data Range: {test_data['Date'].min()} to {test_data['Date'].max()}")

# Normalization (Fit on Train only to prevent leakage)
scaler = MinMaxScaler(feature_range=(0, 1))
train_data['Dengue_Cases_Scaled'] = scaler.fit_transform(train_data[['Dengue_Cases']])
val_data['Dengue_Cases_Scaled'] = scaler.transform(val_data[['Dengue_Cases']])
test_data['Dengue_Cases_Scaled'] = scaler.transform(test_data[['Dengue_Cases']])

# Sequence Creation
def create_sequences(dataset, look_back=3):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3

# Prepare data for LSTM/GRU/RNN (requires 3D input: [samples, time steps, features])
# We use the scaled data
train_scaled = train_data['Dengue_Cases_Scaled'].values.reshape(-1, 1)

# For validation, we need the last 'look_back' points from train
val_input = np.concatenate((train_scaled[-look_back:], val_data['Dengue_Cases_Scaled'].values.reshape(-1, 1)))
# For test, we need the last 'look_back' points from val
test_input = np.concatenate((val_data['Dengue_Cases_Scaled'].values.reshape(-1, 1)[-look_back:], test_data['Dengue_Cases_Scaled'].values.reshape(-1, 1)))

X_train, y_train = create_sequences(train_scaled, look_back)
X_val, y_val = create_sequences(val_input, look_back)
X_test, y_test = create_sequences(test_input, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Convert to TensorFlow Tensors
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
# X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
# y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
# X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Keep as Numpy arrays for now to allow easy slicing with indices
# We will convert to tensors inside the training loop or dataset creation

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Create tf.data.Dataset for batch processing
BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class DengueForecaster:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.lstm_model = self._build_model('LSTM')
        self.gru_model = self._build_model('GRU')
        self.rnn_model = self._build_model('RNN')
        
        # Define loss and separate optimizers for each model
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.lstm_optimizer = tf.keras.optimizers.Adam()
        self.gru_optimizer = tf.keras.optimizers.Adam()
        self.rnn_optimizer = tf.keras.optimizers.Adam()

    def _build_model(self, model_type):
        model = Sequential()
        if model_type == 'LSTM':
            model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
        elif model_type == 'GRU':
            model.add(GRU(50, return_sequences=True, input_shape=self.input_shape))
            model.add(GRU(50))
        elif model_type == 'RNN':
            model.add(SimpleRNN(50, return_sequences=True, input_shape=self.input_shape))
            model.add(SimpleRNN(50))
        
        model.add(Dense(1))
        return model

    @tf.function(reduce_retracing=True)
    def train_step(self, model, optimizer, x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = self.loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(self, model_name, dataset, epochs=50):
        if model_name == 'LSTM':
            model = self.lstm_model
            optimizer = self.lstm_optimizer
        elif model_name == 'GRU':
            model = self.gru_model
            optimizer = self.gru_optimizer
        elif model_name == 'RNN':
            model = self.rnn_model
            optimizer = self.rnn_optimizer
        else:
            raise ValueError("Invalid model name")

        # Build optimizer variables outside of tf.function
        if hasattr(optimizer, 'build'):
            optimizer.build(model.trainable_variables)

        print(f"Training {model_name}...")
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for x_batch, y_batch in dataset:
                loss = self.train_step(model, optimizer, x_batch, y_batch)
                total_loss += loss
                num_batches += 1
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 1], dtype=tf.float32)])
    def predict_lstm(self, x):
        return self.lstm_model(x, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 1], dtype=tf.float32)])
    def predict_gru(self, x):
        return self.gru_model(x, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 1], dtype=tf.float32)])
    def predict_rnn(self, x):
        return self.rnn_model(x, training=False)

# Initialize Forecaster
forecaster = DengueForecaster(input_shape=(look_back, 1))

# --- Generate Meta-Features via Cross-Validation (Stacking) ---
print("Generating Meta-Features for Ensemble Training...")
tscv = TimeSeriesSplit(n_splits=3)
meta_X_cv = []
meta_y_cv = []

# We need to align X_train (numpy) with train_data (DataFrame) for Prophet
# X_train starts at index 'look_back' of train_data
train_df_aligned = train_data.iloc[look_back:].reset_index(drop=True)

for train_idx, val_idx in tscv.split(X_train):
    # Split Data
    X_t, X_v = X_train[train_idx], X_train[val_idx]
    y_t, y_v = y_train[train_idx], y_train[val_idx]
    
    # Create Dataset
    cv_dataset = tf.data.Dataset.from_tensor_slices((X_t, y_t)).batch(16)
    
    # Train Temp Models
    temp_forecaster = DengueForecaster(input_shape=(look_back, 1))
    temp_forecaster.train('LSTM', cv_dataset, epochs=50) # Fewer epochs for speed
    temp_forecaster.train('GRU', cv_dataset, epochs=50)
    temp_forecaster.train('RNN', cv_dataset, epochs=50)
    
    # Train Temp Prophet
    # Get dates for Prophet
    # train_idx are indices into X_train. 
    # X_train[0] corresponds to train_data.iloc[look_back]
    # So we need train_data.iloc[look_back + train_idx]
    
    # Get the subset of the original dataframe corresponding to this fold's training data
    # We need to be careful. Prophet needs 'ds' and 'y'.
    # The 'y' for Prophet should match the target of the LSTM (which is at time t).
    # LSTM input is t-3..t-1, target is t.
    
    # Let's simplify: Just take the dates corresponding to y_t
    # y_t indices in X_train are train_idx.
    # The corresponding dates in train_data are at (look_back + train_idx)
    
    curr_train_dates = train_data.iloc[look_back:].iloc[train_idx]['Date']
    curr_train_y = train_data.iloc[look_back:].iloc[train_idx]['Dengue_Cases']
    
    curr_prophet_df = pd.DataFrame({'ds': curr_train_dates, 'y': curr_train_y})
    
    m = Prophet()
    m.fit(curr_prophet_df)
    
    # Predict on Validation Fold
    # Convert to tensor for prediction
    X_v_tensor = tf.convert_to_tensor(X_v, dtype=tf.float32)
    p_lstm = scaler.inverse_transform(temp_forecaster.predict_lstm(X_v_tensor))
    p_gru = scaler.inverse_transform(temp_forecaster.predict_gru(X_v_tensor))
    p_rnn = scaler.inverse_transform(temp_forecaster.predict_rnn(X_v_tensor))
    
    curr_val_dates = train_data.iloc[look_back:].iloc[val_idx]['Date']
    future_p = pd.DataFrame({'ds': curr_val_dates})
    p_prophet = m.predict(future_p)['yhat'].values.reshape(-1, 1)
    
    # Stack
    min_len_cv = min(len(p_lstm), len(p_prophet))
    
    # Add Month Feature
    fold_months = curr_val_dates.dt.month.values.reshape(-1, 1)
    
    fold_meta = np.column_stack((p_lstm[:min_len_cv], p_gru[:min_len_cv], p_rnn[:min_len_cv], p_prophet[:min_len_cv], fold_months[:min_len_cv]))
    
    meta_X_cv.append(fold_meta)
    meta_y_cv.append(scaler.inverse_transform(y_v.reshape(-1, 1))[:min_len_cv])

meta_X_train_cv = np.vstack(meta_X_cv)
meta_y_train_cv = np.concatenate(meta_y_cv)
print(f"Generated {len(meta_X_train_cv)} meta-training samples from Cross-Validation.")

# Train Models (Full Training)
forecaster.train('LSTM', train_dataset, epochs=150)
forecaster.train('GRU', train_dataset)
forecaster.train('RNN', train_dataset)

# Train Prophet (with Hyperparameter Tuning)
print("Tuning and Training Prophet...")
prophet_data = train_data[['Date', 'Dengue_Cases']].rename(columns={'Date': 'ds', 'Dengue_Cases': 'y'})
val_prophet_data = val_data[['Date', 'Dengue_Cases']].rename(columns={'Date': 'ds', 'Dengue_Cases': 'y'})

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

best_params = None
best_rmse = float('inf')
best_model = None

print("Grid Search for Prophet Parameters...")
for params in ParameterGrid(param_grid):
    try:
        m = Prophet(**params)
        m.fit(prophet_data)
        
        forecast = m.predict(val_prophet_data[['ds']])
        rmse = np.sqrt(mean_squared_error(val_prophet_data['y'], forecast['yhat']))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = m
    except Exception as e:
        continue

print(f"Best Prophet Params: {best_params}")
print(f"Best Prophet Validation RMSE: {best_rmse}")

prophet_model = best_model

print("Base Learners Trained.")

# Generate Validation Predictions
# LSTM, GRU, RNN
lstm_val_pred = scaler.inverse_transform(forecaster.predict_lstm(X_val))
gru_val_pred = scaler.inverse_transform(forecaster.predict_gru(X_val))
rnn_val_pred = scaler.inverse_transform(forecaster.predict_rnn(X_val))

# Prophet
val_dates = val_data[['Date']].rename(columns={'Date': 'ds'})
prophet_val_pred = prophet_model.predict(val_dates)['yhat'].values.reshape(-1, 1)

# Ensure all have same length
min_len = min(len(lstm_val_pred), len(prophet_val_pred))
lstm_val_pred = lstm_val_pred[:min_len]
gru_val_pred = gru_val_pred[:min_len]
rnn_val_pred = rnn_val_pred[:min_len]
prophet_val_pred = prophet_val_pred[:min_len]
y_val_actual = val_data['Dengue_Cases'].values[:min_len]

# Prepare Meta-Features
val_months = val_data['Date'].dt.month.values[:min_len].reshape(-1, 1)
meta_X_val = np.column_stack((lstm_val_pred, gru_val_pred, rnn_val_pred, prophet_val_pred, val_months))
meta_y_val = y_val_actual.reshape(-1, 1)

# Combine CV Meta-Features with Validation Meta-Features
full_meta_X = np.vstack((meta_X_train_cv, meta_X_val))
full_meta_y = np.concatenate((meta_y_train_cv, meta_y_val))

# Train Meta-Model (XGBoost with Tuning)
print("Tuning and Training XGBoost Meta-Model...")
xgb_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

best_xgb_score = float('inf')
best_xgb_model = None

# Simple Grid Search for XGBoost
# We use the Validation set as the 'test' set for this grid search to avoid overfitting to the CV data
# Or better, we use K-Fold on the full_meta_X. 
# Given small size, let's just use a simple loop and validate on meta_X_val (which is the most recent data)

for n_est in xgb_param_grid['n_estimators']:
    for depth in xgb_param_grid['max_depth']:
        for lr in xgb_param_grid['learning_rate']:
            for reg_a in xgb_param_grid['reg_alpha']:
                for reg_l in xgb_param_grid['reg_lambda']:
                    for subsample in xgb_param_grid['subsample']:
                        for colsample in xgb_param_grid['colsample_bytree']:
                            model = xgb.XGBRegressor(objective='reg:squarederror', 
                                                     n_estimators=n_est, 
                                                     max_depth=depth, 
                                                     learning_rate=lr,
                                                     reg_alpha=reg_a,
                                                     reg_lambda=reg_l,
                                                     subsample=subsample,
                                                     colsample_bytree=colsample,
                                                     n_jobs=-1) # Use parallel processing
                            
                            # Train on the CV part (historical), Validate on the Val part (recent)
                            # This mimics the temporal structure
                            model.fit(meta_X_train_cv, meta_y_train_cv)
                            
                            preds = model.predict(meta_X_val)
                            rmse = np.sqrt(mean_squared_error(meta_y_val, preds))
                            
                            if rmse < best_xgb_score:
                                best_xgb_score = rmse
                                best_xgb_model = model

print(f"Best XGBoost RMSE on Validation Set: {best_xgb_score}")
# Retrain best model on ALL available meta-data
best_xgb_model.fit(full_meta_X, full_meta_y)
xgb_model = best_xgb_model

# Generate Test Predictions
# Base Models
lstm_test_pred = scaler.inverse_transform(forecaster.predict_lstm(X_test))
gru_test_pred = scaler.inverse_transform(forecaster.predict_gru(X_test))
rnn_test_pred = scaler.inverse_transform(forecaster.predict_rnn(X_test))

test_dates = test_data[['Date']].rename(columns={'Date': 'ds'})
prophet_test_pred = prophet_model.predict(test_dates)['yhat'].values.reshape(-1, 1)

# Ensure alignment
min_len_test = min(len(lstm_test_pred), len(prophet_test_pred))
lstm_test_pred = lstm_test_pred[:min_len_test]
gru_test_pred = gru_test_pred[:min_len_test]
rnn_test_pred = rnn_test_pred[:min_len_test]
prophet_test_pred = prophet_test_pred[:min_len_test]
y_test_actual = test_data['Dengue_Cases'].values[:min_len_test]

# Meta Features for Test
test_months = test_data['Date'].dt.month.values[:min_len_test].reshape(-1, 1)
meta_X_test = np.column_stack((lstm_test_pred, gru_test_pred, rnn_test_pred, prophet_test_pred, test_months))

# Final Ensemble Predictions
ensemble_test_pred = xgb_model.predict(meta_X_test)

print("Ensemble Predictions Generated.")

# --- Overfitting Check (Generate Train Predictions) ---
lstm_train_pred = scaler.inverse_transform(forecaster.predict_lstm(X_train))
gru_train_pred = scaler.inverse_transform(forecaster.predict_gru(X_train))
rnn_train_pred = scaler.inverse_transform(forecaster.predict_rnn(X_train))

train_dates = train_data[['Date']].rename(columns={'Date': 'ds'}).iloc[look_back:]
prophet_train_pred = prophet_model.predict(train_dates)['yhat'].values.reshape(-1, 1)

# Actual Train Values (Inverse Transform)
# y_train is now a numpy array, not a tensor
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

# Ensure alignment
min_len_train = min(len(lstm_train_pred), len(prophet_train_pred))
lstm_train_pred = lstm_train_pred[:min_len_train]
gru_train_pred = gru_train_pred[:min_len_train]
rnn_train_pred = rnn_train_pred[:min_len_train]
prophet_train_pred = prophet_train_pred[:min_len_train]
y_train_actual = y_train_actual[:min_len_train]

# Meta Features for Train
train_months = train_data['Date'].dt.month.values[look_back:][:min_len_train].reshape(-1, 1)
meta_X_train = np.column_stack((lstm_train_pred, gru_train_pred, rnn_train_pred, prophet_train_pred, train_months))
ensemble_train_pred = xgb_model.predict(meta_X_train)

models_train_preds = {
    'LSTM': lstm_train_pred,
    'GRU': gru_train_pred,
    'RNN': rnn_train_pred,
    'Prophet': prophet_train_pred,
    'Ensemble': ensemble_train_pred
}

# Define Thresholds
threshold = np.percentile(df['Affected'], 75)
medium_threshold = np.percentile(df['Affected'], 50)
print(f"Outbreak Threshold (High Risk - 75th Percentile): {threshold}")
print(f"Medium Risk Threshold (50th Percentile): {medium_threshold}")

def get_class(value, threshold):
    return 1 if value > threshold else 0

# Binarize Actuals
y_test_class = [get_class(y, threshold) for y in y_test_actual]

# Binarize Predictions
models_preds = {
    'LSTM': lstm_test_pred,
    'GRU': gru_test_pred,
    'RNN': rnn_test_pred,
    'Prophet': prophet_test_pred,
    'Ensemble': ensemble_test_pred
}

metrics_list = []

for name, preds in models_preds.items():
    # Flatten preds
    preds_flat = preds.flatten() if hasattr(preds, 'flatten') else preds
    
    # Binarize
    preds_class = [get_class(p, threshold) for p in preds_flat]
    
    # Metrics
    acc = accuracy_score(y_test_class, preds_class)
    prec_macro = precision_score(y_test_class, preds_class, average='macro')
    rec_macro = recall_score(y_test_class, preds_class, average='macro')
    f1_macro = f1_score(y_test_class, preds_class, average='macro')
    
    prec_micro = precision_score(y_test_class, preds_class, average='micro')
    rec_micro = recall_score(y_test_class, preds_class, average='micro')
    f1_micro = f1_score(y_test_class, preds_class, average='micro')
    
    mcc = matthews_corrcoef(y_test_class, preds_class)

    # Calculate RMSE, MAE, WMAPE (Test)
    y_true = np.array(y_test_actual)
    y_pred = np.array(preds_flat)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) != 0 else np.inf
    
    # Calculate Train Metrics
    train_preds = models_train_preds[name]
    train_preds_flat = train_preds.flatten() if hasattr(train_preds, 'flatten') else train_preds
    y_train_flat = y_train_actual.flatten()
    
    train_rmse = np.sqrt(mean_squared_error(y_train_flat, train_preds_flat))
    train_mae = mean_absolute_error(y_train_flat, train_preds_flat)
    train_wmape = np.sum(np.abs(y_train_flat - train_preds_flat)) / np.sum(np.abs(y_train_flat)) if np.sum(np.abs(y_train_flat)) != 0 else np.inf
    
    # Overfitting Indicator (Train RMSE vs Test RMSE)
    # If Test RMSE is significantly higher than Train RMSE (e.g., > 1.5x), it might be overfitting.
    overfit_ratio = rmse / train_rmse if train_rmse > 0 else 0
    is_overfitting = "Yes" if overfit_ratio > 1.5 else "No"

    # AUC (using raw predictions as scores)
    # Normalize scores to 0-1 for AUC if needed, but roc_auc_score handles raw scores.
    # However, we need to handle the case where only one class is present in y_test_class
    try:
        fpr, tpr, _ = roc_curve(y_test_class, preds_flat)
        auc_score = auc(fpr, tpr)
    except:
        auc_score = 0
        
    metrics_list.append({
        'Method': name,
        'Macro Prec': prec_macro, 'Macro Rec': rec_macro, 'Macro F1': f1_macro,
        'Micro Prec': prec_micro, 'Micro Rec': rec_micro, 'Micro F1': f1_micro,
        'MCC': mcc, 'Accuracy': acc, 'AUC': auc_score, 
        'RMSE': rmse, 'MAE': mae, 'WMAPE': wmape,
        'Train RMSE': train_rmse, 'Train MAE': train_mae, 'Train WMAPE': train_wmape,
        'Overfitting?': is_overfitting
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# GRU Investigation on 2025 Data (Test Set)
print("\n--- GRU Model Investigation (2025 Data) ---")
print(f"Test Set Date Range: {test_data['Date'].min()} to {test_data['Date'].max()}")
gru_test_class = [get_class(p, threshold) for p in gru_test_pred]
print("GRU Test Predictions (Class):", gru_test_class)
print("Actual Test Classes:", y_test_class)
if accuracy_score(y_test_class, gru_test_class) == 1.0:
    print("GRU achieved 100% accuracy on 2025 data.")
    print("Investigation: The test set (2025) might have a consistent trend that GRU captured well, or the threshold separation is distinct.")
else:
    print(f"GRU Accuracy on 2025 data: {accuracy_score(y_test_class, gru_test_class)}")

# --- Cross Validation Accuracy (TimeSeriesSplit) ---
print("\n--- Cross Validation Accuracy (Ensemble) ---")
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []
fold = 1

# We need to use the full dataset (scaled) for CV
# X_full and y_full are already created later, but we need them here or recreate them.
# Let's recreate X and y from the whole dataset for CV purposes
cv_scaler = MinMaxScaler(feature_range=(0, 1))
cv_scaled_data = cv_scaler.fit_transform(data[['Dengue_Cases']])
X_cv, y_cv = create_sequences(cv_scaled_data, look_back)
X_cv = np.reshape(X_cv, (X_cv.shape[0], X_cv.shape[1], 1))

# Convert to numpy for indexing
X_cv_np = X_cv
y_cv_np = y_cv

for train_index, test_index in tscv.split(X_cv_np):
    # Further split train into train/val for Meta-Learner
    # We'll take the last 20% of the train_index as validation
    val_split_idx = int(len(train_index) * 0.8)
    cv_train_idx = train_index[:val_split_idx]
    cv_val_idx = train_index[val_split_idx:]
    
    # Prepare Data
    X_train_fold = tf.convert_to_tensor(X_cv_np[cv_train_idx], dtype=tf.float32)
    y_train_fold = tf.convert_to_tensor(y_cv_np[cv_train_idx], dtype=tf.float32)
    X_val_fold = tf.convert_to_tensor(X_cv_np[cv_val_idx], dtype=tf.float32)
    y_val_fold = tf.convert_to_tensor(y_cv_np[cv_val_idx], dtype=tf.float32)
    X_test_fold = tf.convert_to_tensor(X_cv_np[test_index], dtype=tf.float32)
    y_test_fold = y_cv_np[test_index] # Keep as numpy for evaluation
    
    # Create Datasets
    train_ds_fold = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Train Base Models
    cv_forecaster = DengueForecaster(input_shape=(look_back, 1))
    # Reduced epochs for CV to save time
    cv_forecaster.train('LSTM', train_ds_fold, epochs=50) 
    cv_forecaster.train('GRU', train_ds_fold, epochs=50)
    cv_forecaster.train('RNN', train_ds_fold, epochs=50)
    
    # Prophet (Train on cv_train_idx data)
    # We need dates.
    # data index aligns with X_cv_np index shifted by look_back
    # X_cv[i] corresponds to data.iloc[i:i+look_back]
    # y_cv[i] corresponds to data.iloc[i+look_back]
    # So y_cv indices map to data indices: index + look_back
    
    prophet_train_dates = data.iloc[cv_train_idx + look_back][['Date', 'Dengue_Cases']].rename(columns={'Date': 'ds', 'Dengue_Cases': 'y'})
    cv_prophet = Prophet()
    cv_prophet.fit(prophet_train_dates)
    
    # Generate Meta-Features on Validation Fold
    p_lstm = cv_scaler.inverse_transform(cv_forecaster.predict_lstm(X_val_fold))
    p_gru = cv_scaler.inverse_transform(cv_forecaster.predict_gru(X_val_fold))
    p_rnn = cv_scaler.inverse_transform(cv_forecaster.predict_rnn(X_val_fold))
    
    prophet_val_dates = data.iloc[cv_val_idx + look_back][['Date']].rename(columns={'Date': 'ds'})
    p_prophet = cv_prophet.predict(prophet_val_dates)['yhat'].values.reshape(-1, 1)
    
    # Align lengths (Prophet might return different length if dates issue, but should be fine here)
    min_len_fold = min(len(p_lstm), len(p_prophet))
    meta_X_val_fold = np.column_stack((p_lstm[:min_len_fold], p_gru[:min_len_fold], p_rnn[:min_len_fold], p_prophet[:min_len_fold]))
    
    # Actuals for Validation (Inverse Transformed)
    y_val_fold_actual = cv_scaler.inverse_transform(y_cv_np[cv_val_idx].reshape(-1, 1))[:min_len_fold]
    
    # Train Meta-Learner
    cv_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    cv_xgb.fit(meta_X_val_fold, y_val_fold_actual)
    
    # Predict on Test Fold
    t_lstm = cv_scaler.inverse_transform(cv_forecaster.predict_lstm(X_test_fold))
    t_gru = cv_scaler.inverse_transform(cv_forecaster.predict_gru(X_test_fold))
    t_rnn = cv_scaler.inverse_transform(cv_forecaster.predict_rnn(X_test_fold))
    
    prophet_test_dates = data.iloc[test_index + look_back][['Date']].rename(columns={'Date': 'ds'})
    t_prophet = cv_prophet.predict(prophet_test_dates)['yhat'].values.reshape(-1, 1)
    
    min_len_test_fold = min(len(t_lstm), len(t_prophet))
    meta_X_test_fold = np.column_stack((t_lstm[:min_len_test_fold], t_gru[:min_len_test_fold], t_rnn[:min_len_test_fold], t_prophet[:min_len_test_fold]))
    
    # Ensemble Prediction
    fold_preds = cv_xgb.predict(meta_X_test_fold)
    
    # Evaluate Accuracy
    y_test_fold_actual = cv_scaler.inverse_transform(y_test_fold.reshape(-1, 1))[:min_len_test_fold]
    
    # Binarize
    fold_preds_class = [get_class(p, threshold) for p in fold_preds]
    y_test_fold_class = [get_class(y, threshold) for y in y_test_fold_actual]
    
    fold_acc = accuracy_score(y_test_fold_class, fold_preds_class)
    cv_scores.append(fold_acc)
    print(f"Fold {fold} Accuracy: {fold_acc:.4f}")
    fold += 1

print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Highlight best values (Visual inspection in notebook)
# In a real report, we would bold them. Here we just print the table.

# Model Comparison Graph
plt.figure(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Macro F1', 'AUC', 'MCC']
metrics_melted = metrics_df.melt(id_vars='Method', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

sns.barplot(x='Metric', y='Score', hue='Method', data=metrics_melted, palette='viridis')
plt.title('Model Performance Comparison')
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
# plt.show()

# Training vs Testing Performance Comparison
plt.figure(figsize=(12, 6))
train_test_metrics = metrics_df[['Method', 'RMSE', 'Train RMSE']].melt(id_vars='Method', var_name='Dataset', value_name='RMSE_Value')
sns.barplot(x='Method', y='RMSE_Value', hue='Dataset', data=train_test_metrics, palette='Set2')
plt.title('Training vs Testing Performance Comparison (RMSE)')
plt.ylabel('RMSE (Lower is Better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('train_test_comparison.png')
# plt.show()

from sklearn.cluster import DBSCAN

# Coordinates for Divisions (Approximate)
division_coords = {
    'Barishal': (22.7010, 90.3535),
    'Chattogram': (22.3569, 91.7832),
    'Dhaka': (23.8103, 90.4125),
    'Khulna': (22.8456, 89.5403),
    'Mymensingh': (24.7471, 90.4203),
    'Rajshahi': (24.3636, 88.6241),
    'Rangpur': (25.7439, 89.2752),
    'Sylhet': (24.8949, 91.8687)
}

# Aggregate cases by district (Total over the period)
district_cols = [col for col in df.columns if '(Affected)' in col]
district_totals = df[district_cols].sum()
district_names = [col.replace('(Affected)', '') for col in district_cols]

spatial_data = pd.DataFrame({
    'District': district_names,
    'Cases': district_totals.values,
    'Lat': [division_coords[d][0] for d in district_names],
    'Lon': [division_coords[d][1] for d in district_names]
})

print(spatial_data)

# DBSCAN Clustering
# We cluster based on location. Weighting by cases is tricky in standard DBSCAN.
# Usually, we look for spatial clusters of high cases.
# Here we just cluster the locations and maybe visualize intensity.
coords = spatial_data[['Lat', 'Lon']].values
db = DBSCAN(eps=1.5, min_samples=2).fit(coords)
spatial_data['Cluster'] = db.labels_

# Moran's I Calculation
def calculate_morans_i(values, coords):
    N = len(values)
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                w[i][j] = 1 / (dist + 1e-6) # Inverse distance weight
    
    W = np.sum(w)
    mean_val = np.mean(values)
    
    num = 0
    den = 0
    for i in range(N):
        den += (values[i] - mean_val)**2
        for j in range(N):
            num += w[i][j] * (values[i] - mean_val) * (values[j] - mean_val)
            
    return (N / W) * (num / den)

moran_i = abs(calculate_morans_i(spatial_data['Cases'].values, coords))
print(f"Moran's I: {moran_i}")

# Visuals (Scatter Plot)
plt.figure(figsize=(10, 8))
plt.scatter(spatial_data['Lon'], spatial_data['Lat'], s=spatial_data['Cases']/100, c=spatial_data['Cluster'], cmap='viridis', alpha=0.6)
for i, txt in enumerate(spatial_data['District']):
    plt.annotate(txt, (spatial_data['Lon'][i], spatial_data['Lat'][i]))
plt.title('Spatial Clustering of Dengue Cases (Size = Cases)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('spatial_clustering.png')
# plt.show()

# Retrain on Full Dataset
# Fit scaler on full data for final training
scaler_full = MinMaxScaler(feature_range=(0, 1))
full_scaled = scaler_full.fit_transform(data[['Dengue_Cases']])
scaler = scaler_full # Update scaler reference

X_full, y_full = create_sequences(full_scaled, look_back)
X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

# Convert to Tensor
X_full = tf.convert_to_tensor(X_full, dtype=tf.float32)
y_full = tf.convert_to_tensor(y_full, dtype=tf.float32)

# Create Dataset
full_dataset = tf.data.Dataset.from_tensor_slices((X_full, y_full)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Initialize Forecaster for Full Data
full_forecaster = DengueForecaster(input_shape=(look_back, 1))

# Retrain Base Models
full_forecaster.train('LSTM', full_dataset, epochs=150)
full_forecaster.train('GRU', full_dataset)
full_forecaster.train('RNN', full_dataset)

prophet_full = Prophet()
prophet_full.fit(data[['Date', 'Dengue_Cases']].rename(columns={'Date': 'ds', 'Dengue_Cases': 'y'}))

# Generate Meta Features for Full Data
lstm_full_pred = scaler.inverse_transform(full_forecaster.predict_lstm(X_full))
gru_full_pred = scaler.inverse_transform(full_forecaster.predict_gru(X_full))
rnn_full_pred = scaler.inverse_transform(full_forecaster.predict_rnn(X_full))

# Prophet Predictions for Full Data
prophet_full_pred = prophet_full.predict(data[['Date']].rename(columns={'Date': 'ds'}).iloc[look_back:])['yhat'].values.reshape(-1, 1)

# Align
min_len_full = min(len(lstm_full_pred), len(prophet_full_pred))
lstm_full_pred = lstm_full_pred[:min_len_full]
gru_full_pred = gru_full_pred[:min_len_full]
rnn_full_pred = rnn_full_pred[:min_len_full]
prophet_full_pred = prophet_full_pred[:min_len_full]
y_full_actual = data['Dengue_Cases'].values[look_back:][:min_len_full]

# Add Month Feature
full_months = data['Date'].dt.month.values[look_back:][:min_len_full].reshape(-1, 1)

meta_X_full = np.column_stack((lstm_full_pred, gru_full_pred, rnn_full_pred, prophet_full_pred, full_months))
meta_y_full = y_full_actual

# Retrain XGBoost (Use Best Params from Tuning)
xgb_full = xgb.XGBRegressor(**best_xgb_model.get_params())
xgb_full.fit(meta_X_full, meta_y_full)

# Recursive Forecasting for Next 6 Months
future_months = 6
last_sequence = full_scaled[-look_back:]
future_predictions_ensemble = []
future_predictions_rnn = []
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_months+1, freq='M')[1:]

current_seq = last_sequence.reshape(1, look_back, 1)

for i in range(future_months):
    # Convert current_seq to tensor for prediction
    current_seq_tensor = tf.convert_to_tensor(current_seq, dtype=tf.float32)

    # Base Predictions
    p_lstm = scaler.inverse_transform(full_forecaster.predict_lstm(current_seq_tensor))[0][0]
    p_gru = scaler.inverse_transform(full_forecaster.predict_gru(current_seq_tensor))[0][0]
    p_rnn = scaler.inverse_transform(full_forecaster.predict_rnn(current_seq_tensor))[0][0]
    
    # Prophet Prediction
    next_date = future_dates[i]
    p_prophet = prophet_full.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].values[0]
    
    # Ensemble Prediction
    # Add Month Feature
    next_month = next_date.month
    meta_input = np.array([[p_lstm, p_gru, p_rnn, p_prophet, next_month]])
    final_pred = xgb_full.predict(meta_input)[0]
    
    future_predictions_ensemble.append(final_pred)
    future_predictions_rnn.append(p_rnn)
    
    # Update Sequence
    # We use RNN prediction for feedback as it is more precise (lower MAE/RMSE)
    # This aligns with "Trust RNN for absolute case predictions"
    feedback_value = p_rnn 
    feedback_scaled = scaler.transform([[feedback_value]])[0][0]
    current_seq = np.append(current_seq[:, 1:, :], [[[feedback_scaled]]], axis=1)

print("Future Predictions (RNN - Used for Planning):", future_predictions_rnn)
print("Future Predictions (Ensemble - Used for Risk):", future_predictions_ensemble)

# Classify Future Risk (Using Ensemble)
def classify_risk(p, high_th, med_th):
    if p > high_th:
        return 'High'
    elif p > med_th:
        return 'Medium'
    else:
        return 'Low'

future_risk = [classify_risk(p, threshold, medium_threshold) for p in future_predictions_ensemble]
print("Future Risk Levels:", future_risk)

# Constants
SEVERITY_RATE = 0.20 # 20% of cases are severe
BAGS_PER_PATIENT = 2
SAFETY_BUFFER = 50 # Bags

# Calculate National Demand (Using RNN Predictions)
national_demand = []
for pred in future_predictions_rnn:
    demand = (pred * SEVERITY_RATE * BAGS_PER_PATIENT) + SAFETY_BUFFER
    national_demand.append(demand)

print("National Blood Demand (Next 6 Months):", national_demand)

# District Shares
total_cases_all = district_totals.sum()
district_shares = district_totals / total_cases_all
district_shares.index = district_names # Fix index to match district names

# Forecast Cases by District
district_cases_forecast = pd.DataFrame(index=district_names)
for i, national_cases in enumerate(future_predictions_rnn):
    district_cases_forecast[f'Month_{i+1}'] = district_shares * national_cases

# Allocation
allocation_df = pd.DataFrame(index=district_names)
for i, month_demand in enumerate(national_demand):
    allocation_df[f'Month_{i+1}'] = district_shares * month_demand

# Hotspot Adjustment (+10% for hotspots)
# We use the clusters found in Phase 6
hotspots = spatial_data[spatial_data['Cluster'] != -1]['District'].tolist()
print("Identified Hotspots:", hotspots)

for district in hotspots:
    if district in allocation_df.index:
        allocation_df.loc[district] *= 1.10
    if district in district_cases_forecast.index:
        district_cases_forecast.loc[district] *= 1.10

print("Future Dengue Cases by District (Forecast):")
print(district_cases_forecast)

print("Blood Allocation Plan (Bags):")
print(allocation_df)

print("Blood Allocation Plan (Bags):")
print(allocation_df)

# Figure 1: Historical Trend
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Dengue_Cases'], label='Actual Cases')
plt.title('Figure 1: Historical Trend of Dengue Cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.savefig('figure_1_historical_trend.png')
# plt.show()

# Figure 2: Actual vs. Predicted (Test Set)
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'], y_test_actual, label='Actual')
plt.plot(test_data['Date'], ensemble_test_pred, label='Predicted (Ensemble)', linestyle='--')
plt.title('Figure 2: Actual vs. Predicted (Test Set)')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.savefig('figure_2_actual_vs_predicted.png')
# plt.show()

# Figure 3: ROC Curve for All Models
plt.figure(figsize=(10, 8))
for name, preds in models_preds.items():
    # Flatten preds if necessary
    preds_flat = preds.flatten() if hasattr(preds, 'flatten') else preds
    
    # Calculate ROC and AUC
    try:
        fpr, tpr, _ = roc_curve(y_test_class, preds_flat)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    except Exception as e:
        print(f"Could not plot ROC for {name}: {e}")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 3: ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig('figure_3_roc_curve.png')
# plt.show()

# Figure 4: Confusion Matrix
cm = confusion_matrix(y_test_class, [get_class(p, threshold) for p in ensemble_test_pred])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Figure 4: Confusion Matrix (Ensemble)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('figure_4_confusion_matrix.png')
# plt.show()

# Figure 5: Spatial Hotspots (Already plotted in Phase 6, repeating for completeness)
plt.figure(figsize=(10, 8))
plt.scatter(spatial_data['Lon'], spatial_data['Lat'], s=spatial_data['Cases']/100, c=spatial_data['Cluster'], cmap='viridis', alpha=0.6)
for i, txt in enumerate(spatial_data['District']):
    plt.annotate(txt, (spatial_data['Lon'][i], spatial_data['Lat'][i]))
plt.title('Figure 5: Spatial Hotspots (DBSCAN)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('figure_5_spatial_hotspots.png')
# plt.show()

# Figure 6: Future Forecast
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Dengue_Cases'], label='Historical')
plt.plot(future_dates, future_predictions_rnn, label='Forecast (RNN - Planning)', linestyle='--', color='red')
plt.plot(future_dates, future_predictions_ensemble, label='Forecast (Ensemble - Risk)', linestyle=':', color='blue')
plt.title('Figure 6: Future Forecast (Next 6 Months)')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.savefig('figure_6_future_forecast.png')
# plt.show()

# Figure 7: Blood Demand Bar Chart
plt.figure(figsize=(10, 6))
colors = ['red' if r == 'High' else 'green' for r in future_risk]
plt.bar(range(1, 7), national_demand, color=colors)
plt.title('Figure 7: Estimated National Blood Demand')
plt.xlabel('Future Month')
plt.ylabel('Bags')
plt.xticks(range(1, 7), [f'M{i}' for i in range(1, 7)])
plt.savefig('figure_7_blood_demand.png')
# plt.show()

# Figure 8: Regional Allocation (Month 1)
plt.figure(figsize=(12, 6))
allocation_df['Month_1'].plot(kind='bar')
plt.title('Figure 8: Regional Blood Allocation (Month 1)')
plt.ylabel('Bags')
plt.xlabel('District')
plt.savefig('figure_8_regional_allocation.png')
# plt.show()

# Figure 9: Predicted District Cases (Month 1)
plt.figure(figsize=(12, 6))
district_cases_forecast['Month_1'].plot(kind='bar', color='orange')
plt.title('Figure 9: Predicted District Cases (Month 1)')
plt.ylabel('Cases')
plt.xlabel('District')
plt.savefig('figure_9_district_cases.png')
# plt.show()
