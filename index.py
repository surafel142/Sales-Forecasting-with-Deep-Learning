import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = 30  # Number of previous days to use for prediction
        self.sequence_length = 30
        self.forecast_days = 30
        
    def generate_realistic_sales_data(self, days=1000):
        """Generate realistic time-series sales data with trends, seasonality, and noise"""
        print("📊 Generating realistic sales dataset...")
        
        np.random.seed(42)
        
        # Create date range
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(days)]
        
        # Base components
        t = np.arange(days)
        
        # 1. Long-term trend (gradual growth)
        trend = 100 + 0.5 * t
        
        # 2. Seasonal patterns
        # Weekly seasonality (7-day cycle)
        weekly_seasonality = 20 * np.sin(2 * np.pi * t / 7)
        
        # Monthly seasonality (30-day cycle)
        monthly_seasonality = 30 * np.sin(2 * np.pi * t / 30)
        
        # Yearly seasonality (365-day cycle)
        yearly_seasonality = 50 * np.sin(2 * np.pi * t / 365)
        
        # 3. Special events (holidays, promotions)
        events = np.zeros(days)
        # Add quarterly peaks
        for i in range(0, days, 90):
            if i + 5 < days:
                events[i:i+5] = np.random.normal(80, 20, 5)
        
        # 4. Random noise
        noise = np.random.normal(0, 15, days)
        
        # 5. External factors (weekends, special days)
        weekend_effect = np.array([15 if (start_date + timedelta(days=x)).weekday() >= 5 else 0 for x in range(days)])
        
        # Combine all components
        sales = (trend + weekly_seasonality + monthly_seasonality + 
                yearly_seasonality + events + noise + weekend_effect)
        
        # Ensure positive sales
        sales = np.maximum(10, sales)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
            'is_holiday': [1 if d.month == 12 and d.day in [24, 25, 31] else 0 for d in dates]
        })
        
        # Add some additional features
        self.df['rolling_avg_7'] = self.df['sales'].rolling(window=7).mean()
        self.df['rolling_avg_30'] = self.df['sales'].rolling(window=30).mean()
        self.df['sales_lag1'] = self.df['sales'].shift(1)
        self.df['sales_lag7'] = self.df['sales'].shift(7)
        
        print(f"✅ Generated {days} days of sales data")
        return self.df
    
    def explore_sales_data(self):
        """Comprehensive exploratory data analysis"""
        print("\n🔍 EXPLORING SALES DATA")
        print("=" * 40)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"\nFirst 10 rows:")
        print(self.df.head(10))
        
        print(f"\n📈 Statistical Summary:")
        print(self.df[['sales', 'rolling_avg_7', 'rolling_avg_30']].describe())
        
        # Sales patterns by day of week
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print("\n📊 Sales by Day of Week:")
for day, sales in self.df.groupby('day_of_week')['sales'].mean().items():
    print(f"  {days[day]}: {sales:.2f}")
        
        # Monthly patterns
        print(f"\n📅 Monthly Sales Patterns:")
        monthly_sales = self.df.groupby('month')['sales'].mean()
        for month, sales in monthly_sales.items():
            print(f"  Month {month}: {sales:.2f}")
    
    def visualize_sales_patterns(self):
        """Create comprehensive sales visualizations"""
        print("\n📊 VISUALIZING SALES PATTERNS")
        print("=" * 40)
        
        plt.figure(figsize=(16, 12))
        
        # 1. Overall sales trend
        plt.subplot(3, 2, 1)
        plt.plot(self.df['date'], self.df['sales'], alpha=0.7, linewidth=1, label='Daily Sales')
        plt.plot(self.df['date'], self.df['rolling_avg_7'], linewidth=2, label='7-Day Moving Avg', color='red')
        plt.plot(self.df['date'], self.df['rolling_avg_30'], linewidth=2, label='30-Day Moving Avg', color='orange')
        plt.title('Sales Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Sales distribution
        plt.subplot(3, 2, 2)
        plt.hist(self.df['sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df['sales'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["sales"].mean():.2f}')
        plt.xlabel('Sales')
        plt.ylabel('Frequency')
        plt.title('Sales Distribution')
        plt.legend()
        
        # 3. Day of week analysis
        plt.subplot(3, 2, 3)
        daily_sales = self.df.groupby('day_of_week')['sales'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = ['blue'] * 5 + ['red', 'red']  # Weekends in red
        plt.bar(days, daily_sales.values, color=colors, alpha=0.7)
        plt.title('Average Sales by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Sales')
        
        # 4. Monthly patterns
        plt.subplot(3, 2, 4)
        monthly_sales = self.df.groupby('month')['sales'].mean()
        plt.bar(monthly_sales.index, monthly_sales.values, color='green', alpha=0.7)
        plt.title('Average Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        
        # 5. Autocorrelation plot
        plt.subplot(3, 2, 5)
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df['sales'].dropna())
        plt.title('Sales Autocorrelation')
        
        # 6. Seasonal decomposition (simplified)
        plt.subplot(3, 2, 6)
        # Show last 90 days for clarity
        recent_data = self.df.tail(90)
        plt.plot(recent_data['date'], recent_data['sales'], label='Actual')
        plt.plot(recent_data['date'], recent_data['rolling_avg_7'], label='Trend (7-day MA)')
        plt.title('Recent Sales Pattern (Last 90 Days)')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, test_size=0.2):
        """Prepare data for LSTM model"""
        print("\n🤖 PREPARING DATA FOR LSTM")
        print("=" * 35)
        
        # Use sales data
        sales_data = self.df['sales'].values.reshape(-1, 1)
        
        # Scale the data
        sales_scaled = self.scaler.fit_transform(sales_data)
        
        # Create sequences
        X, y = self.create_sequences(sales_scaled, self.sequence_length)
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training sequences: {X_train.shape[0]}")
        print(f"Test sequences: {X_test.shape[0]}")
        print(f"Sequence shape: {X_train.shape[1:]")
        
        return X_train, X_test, y_train, y_test, split_idx
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network model"""
        print("\n🧠 BUILDING LSTM NEURAL NETWORK")
        print("=" * 40)
        
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(100, return_sequences=True, input_shape=input_shape,
                       dropout=0.2, recurrent_dropout=0.2),
            
            # Second LSTM layer
            layers.LSTM(80, return_sequences=True, 
                       dropout=0.2, recurrent_dropout=0.2),
            
            # Third LSTM layer
            layers.LSTM(60, return_sequences=False,
                       dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)  # Output layer for regression
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ LSTM Architecture:")
        model.summary()
        
        return model
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test, epochs=100):
        """Train the LSTM model"""
        print("\n🎯 TRAINING LSTM MODEL")
        print("=" * 30)
        
        self.model = self.build_lstm_model(X_train.shape[1:])
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=20, 
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=10, 
                factor=0.5,
                monitor='val_loss'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed!")
        return history
    
    def evaluate_model(self, X_test, y_test, split_idx):
        """Evaluate model performance"""
        print("\n📈 MODEL EVALUATION")
        print("=" * 25)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        
        print(f"📊 Forecasting Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return y_pred, y_test_original, {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("\n📉 PLOTTING TRAINING HISTORY")
        print("=" * 35)
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE During Training')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test_original, y_pred, split_idx):
        """Plot predictions vs actual values"""
        print("\n📊 PREDICTIONS VS ACTUAL VALUES")
        print("=" * 40)
        
        # Create date indices for test set
        test_dates = self.df['date'].iloc[split_idx + self.sequence_length:split_idx + self.sequence_length + len(y_test_original)]
        
        plt.figure(figsize=(15, 10))
        
        # 1. Test set predictions
        plt.subplot(2, 2, 1)
        plt.plot(test_dates, y_test_original, label='Actual Sales', alpha=0.7, linewidth=2)
        plt.plot(test_dates, y_pred, label='Predicted Sales', alpha=0.7, linewidth=2)
        plt.title('LSTM Predictions vs Actual (Test Set)')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_test_original, y_pred, alpha=0.6)
        plt.plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        
        # 3. Residuals
        plt.subplot(2, 2, 3)
        residuals = y_test_original.flatten() - y_pred.flatten()
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 4. Error distribution
        plt.subplot(2, 2, 4)
        errors = np.abs(residuals)
        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean Error: {np.mean(errors):.2f}')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, days=30):
        """Forecast future sales"""
        print(f"\n🔮 FORECASTING NEXT {days} DAYS")
        print("=" * 40)
        
        # Get the last sequence from the data
        last_sequence = self.df['sales'].values[-self.sequence_length:].reshape(-1, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred_scaled = self.model.predict(X_pred, verbose=0)
            forecasts.append(next_pred_scaled[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred_scaled[0, 0]
        
        # Inverse transform forecasts
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts_original = self.scaler.inverse_transform(forecasts)
        
        # Create future dates
        last_date = self.df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(days)]
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecasts_original.flatten(),
            'confidence_lower': forecasts_original.flatten() * 0.9,  # 10% lower
            'confidence_upper': forecasts_original.flatten() * 1.1   # 10% upper
        })
        
        return forecast_df
    
    def plot_forecast(self, forecast_df, show_last=180):
        """Plot the forecast along with historical data"""
        print("\n📈 PLOTTING SALES FORECAST")
        print("=" * 35)
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data (last show_last days)
        historical_data = self.df.tail(show_last)
        
        plt.plot(historical_data['date'], historical_data['sales'], 
                label='Historical Sales', color='blue', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast_df['date'], forecast_df['forecast'], 
                label='Forecast', color='red', linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(forecast_df['date'], 
                        forecast_df['confidence_lower'], 
                        forecast_df['confidence_upper'],
                        alpha=0.3, color='red', label='Confidence Interval')
        
        # Add vertical line separating history and forecast
        last_historical_date = historical_data['date'].iloc[-1]
        plt.axvline(x=last_historical_date, color='black', linestyle='--', 
                   label='Forecast Start')
        
        plt.title(f'Sales Forecast - Next {len(forecast_df)} Days')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print forecast summary
        print(f"\n📊 FORECAST SUMMARY (Next {len(forecast_df)} Days):")
        print(f"  Average Forecast: {forecast_df['forecast'].mean():.2f}")
        print(f"  Maximum Forecast: {forecast_df['forecast'].max():.2f}")
        print(f"  Minimum Forecast: {forecast_df['forecast'].min():.2f}")
        print(f"  Forecast Trend: {'↑ Increasing' if forecast_df['forecast'].iloc[-1] > forecast_df['forecast'].iloc[0] else '↓ Decreasing'}")
    
    def run_complete_analysis(self):
        """Run the complete sales forecasting pipeline"""
        print("🚀 SALES FORECASTING WITH LSTM NEURAL NETWORKS")
        print("=" * 55)
        
        # Step 1: Generate and explore data
        self.generate_realistic_sales_data(1000)
        self.explore_sales_data()
        
        # Step 2: Visualize sales patterns
        self.visualize_sales_patterns()
        
        # Step 3: Prepare LSTM data
        X_train, X_test, y_train, y_test, split_idx = self.prepare_lstm_data()
        
        # Step 4: Train LSTM model
        history = self.train_lstm_model(X_train, y_train, X_test, y_test)
        
        # Step 5: Plot training history
        self.plot_training_history(history)
        
        # Step 6: Evaluate model
        y_pred, y_test_original, metrics = self.evaluate_model(X_test, y_test, split_idx)
        
        # Step 7: Plot predictions
        self.plot_predictions(y_test_original, y_pred, split_idx)
        
        # Step 8: Generate forecast
        forecast_df = self.forecast_future(days=30)
        
        # Step 9: Plot forecast
        self.plot_forecast(forecast_df)
        
        print("\n" + "=" * 50)
        print("✅ SALES FORECASTING PROJECT COMPLETED!")
        print("=" * 50)
        
        return forecast_df

# Advanced Analysis Functions
class AdvancedSalesAnalysis:
    """Advanced sales analysis and model comparison"""
    
    @staticmethod
    def compare_models(predictor):
        """Compare LSTM with traditional methods"""
        print("\n⚡ MODEL COMPARISON: LSTM vs Traditional Methods")
        print("=" * 50)
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Prepare data
        sales_data = predictor.df['sales'].dropna()
        train_size = int(len(sales_data) * 0.8)
        train, test = sales_data[:train_size], sales_data[train_size:]
        
        # Traditional methods
        methods = {}
        
        # 1. Moving Average
        ma_predictions = train.rolling(window=30).mean().iloc[-len(test):].values
        methods['Moving Average'] = ma_predictions
        
        # 2. Exponential Smoothing
        try:
            model_es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
            fitted_es = model_es.fit()
            es_predictions = fitted_es.forecast(len(test))
            methods['Exponential Smoothing'] = es_predictions
        except:
            print("  Exponential Smoothing failed - using naive forecast")
            methods['Exponential Smoothing'] = [train.iloc[-1]] * len(test)
        
        # 3. LSTM (already trained)
        lstm_predictions = predictor.model.predict(predictor.X_test)
        lstm_predictions = predictor.scaler.inverse_transform(lstm_predictions).flatten()
        methods['LSTM'] = lstm_predictions[:len(test)]
        
        # Compare metrics
        comparison = []
        for method_name, predictions in methods.items():
            if len(predictions) == len(test):
                mae = mean_absolute_error(test, predictions)
                rmse = np.sqrt(mean_squared_error(test, predictions))
                comparison.append({
                    'Method': method_name,
                    'MAE': mae,
                    'RMSE': rmse
                })
        
        comparison_df = pd.DataFrame(comparison).sort_values('MAE')
        print("\n📊 Model Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        for i, (method_name, predictions) in enumerate(methods.items()):
            if len(predictions) == len(test):
                plt.plot(test.index[:100], test.values[:100], 
                        label='Actual', color='black', linewidth=2) if i == 0 else None
                plt.plot(test.index[:100], predictions[:100], 
                        label=method_name, alpha=0.7, linewidth=2)
        
        plt.title('Model Comparison: Sales Forecasting')
        plt.xlabel('Time Index')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

# Run the complete project
if __name__ == "__main__":
    # Initialize and run the complete project
    forecaster = SalesForecaster()
    forecast_results = forecaster.run_complete_analysis()
    
    # Run advanced analysis
    advanced_analysis = AdvancedSalesAnalysis()
    advanced_analysis.compare_models(forecaster)

# Interactive Forecasting Function
def interactive_forecasting(forecaster):
    """Allow interactive sales forecasting"""
    print("\n💬 INTERACTIVE SALES FORECASTING")
    print("=" * 45)
    
    while True:
        try:
            print("\nChoose forecasting option:")
            print("1. Forecast next 7 days")
            print("2. Forecast next 30 days")
            print("3. Custom forecast period")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                days = 7
            elif choice == '2':
                days = 30
            elif choice == '3':
                days = int(input("Enter number of days to forecast: "))
            elif choice == '4':
                break
            else:
                print("❌ Invalid choice!")
                continue
            
            # Generate forecast
            forecast_df = forecaster.forecast_future(days=days)
            
            print(f"\n📈 {days}-DAY SALES FORECAST:")
            print("-" * 40)
            for _, row in forecast_df.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['forecast']:.2f} "
                      f"(Range: {row['confidence_lower']:.2f} - {row['confidence_upper']:.2f})")
            
            # Show summary statistics
            avg_forecast = forecast_df['forecast'].mean()
            trend = "↑ Increasing" if forecast_df['forecast'].iloc[-1] > forecast_df['forecast'].iloc[0] else "↓ Decreasing"
            
            print(f"\n📊 Forecast Summary:")
            print(f"  Average: {avg_forecast:.2f}")
            print(f"  Trend: {trend}")
            print(f"  Total Forecasted Sales: {forecast_df['forecast'].sum():.2f}")
            
            # Plot the forecast
            forecaster.plot_forecast(forecast_df)
            
        except ValueError:
            print("❌ Please enter valid numbers!")
        except Exception as e:
            print(f"❌ Error: {e}")

# Uncomment to run interactive forecasting
# interactive_forecasting(forecaster)