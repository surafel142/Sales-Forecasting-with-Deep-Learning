import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class AdvancedSalesAnalysis:
    """Advanced sales analysis and model comparison"""

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.comparison_results = None

    def compare_models(self):
        """Compare LSTM with traditional forecasting methods"""
        print("\n⚡ MODEL COMPARISON: LSTM vs Traditional Methods")
        print("=" * 50)

        sales_data = self.forecaster.df['sales'].dropna()
        train_size = int(len(sales_data) * 0.8)
        train, test = sales_data[:train_size], sales_data[train_size:]

        methods = {}

        # 1. Naive Forecast (last value)
        naive_predictions = [train.iloc[-1]] * len(test)
        methods['Naive'] = naive_predictions

        # 2. Moving Average
        ma_predictions = train.rolling(window=30).mean().iloc[-len(test):].values
        methods['Moving Average'] = ma_predictions

        # 3. Exponential Smoothing
        try:
            model_es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
            fitted_es = model_es.fit()
            es_predictions = fitted_es.forecast(len(test))
            methods['Exponential Smoothing'] = es_predictions
        except Exception as e:
            print(f"  Exponential Smoothing failed: {e}")
            methods['Exponential Smoothing'] = naive_predictions

        # 4. ARIMA
        try:
            model_arima = ARIMA(train, order=(2,1,2))
            fitted_arima = model_arima.fit()
            arima_predictions = fitted_arima.forecast(len(test))
            methods['ARIMA'] = arima_predictions
        except Exception as e:
            print(f"  ARIMA failed: {e}")
            methods['ARIMA'] = naive_predictions

        # 5. LSTM
        lstm_predictions = self.forecaster.model.predict(self.forecaster.X_test)
        lstm_predictions = self.forecaster.scaler.inverse_transform(lstm_predictions).flatten()
        methods['LSTM'] = lstm_predictions[:len(test)]

        # Calculate metrics
        comparison = []
        for method_name, predictions in methods.items():
            if len(predictions) == len(test):
                mae = mean_absolute_error(test, predictions)
                rmse = np.sqrt(mean_squared_error(test, predictions))
                mape = np.mean(np.abs((test - predictions) / test)) * 100
                
                comparison.append({
                    'Method': method_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })

        self.comparison_results = pd.DataFrame(comparison).sort_values('MAE')
        
        print("\n📊 Model Comparison Results:")
        print(self.comparison_results.to_string(index=False))

        # Save comparison results
        self.comparison_results.to_csv('outputs/model_performance/model_comparison.csv', index=False)

        # Plot comparison
        self._plot_model_comparison(methods, test)
        
        return self.comparison_results

    def _plot_model_comparison(self, methods, test):
        """Plot comparison of different forecasting methods"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time series comparison
        plt.subplot(2, 1, 1)
        plt.plot(test.index[:100], test.values[:100], 
                label='Actual', color='black', linewidth=3, alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (method_name, predictions) in enumerate(methods.items()):
            if len(predictions) == len(test):
                plt.plot(test.index[:100], predictions[:100], 
                        label=method_name, color=colors[i % len(colors)], 
                        alpha=0.7, linewidth=2)
        
        plt.title('Model Comparison: Sales Forecasting Performance')
        plt.xlabel('Time Index')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Performance metrics comparison
        plt.subplot(2, 1, 2)
        metrics_plot = self.comparison_results.melt(id_vars=['Method'], 
                                                   value_vars=['MAE', 'RMSE', 'MAPE'],
                                                   var_name='Metric', value_name='Value')
        
        sns.barplot(data=metrics_plot, x='Method', y='Value', hue='Metric')
        plt.title('Forecasting Metrics Comparison')
        plt.xlabel('Forecasting Method')
        plt.ylabel('Error Value')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def residual_analysis(self):
        """Perform detailed residual analysis"""
        print("\n🔍 RESIDUAL ANALYSIS")
        print("=" * 25)
        
        # Get LSTM predictions
        y_pred_scaled = self.forecaster.model.predict(self.forecaster.X_test)
        y_pred = self.forecaster.scaler.inverse_transform(y_pred_scaled)
        y_test_original = self.forecaster.scaler.inverse_transform(
            self.forecaster.y_test.reshape(-1, 1)
        )
        
        residuals = y_test_original.flatten() - y_pred.flatten()
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Residuals over time
        plt.subplot(2, 3, 1)
        plt.plot(residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residuals Over Time')
        plt.xlabel('Time Index')
        plt.ylabel('Residuals')
        
        # Plot 2: Residual distribution
        plt.subplot(2, 3, 2)
        plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(residuals), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(residuals):.2f}')
        plt.title('Residual Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 3: Q-Q plot
        plt.subplot(2, 3, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality Check)')
        
        # Plot 4: Residuals vs Predicted
        plt.subplot(2, 3, 4)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        
        # Plot 5: ACF of residuals
        plt.subplot(2, 3, 5)
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(residuals)
        plt.title('Autocorrelation of Residuals')
        
        # Plot 6: Residuals statistical summary
        plt.subplot(2, 3, 6)
        residual_stats = {
            'Mean': np.mean(residuals),
            'Std Dev': np.std(residuals),
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals)
        }
        
        plt.bar(residual_stats.keys(), residual_stats.values(), 
               color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        plt.title('Residual Statistics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistical tests
        print("\n📊 Residual Statistical Tests:")
        print(f"  Jarque-Bera Normality Test: p-value = {stats.jarque_bera(residuals)[1]:.4f}")
        print(f"  Residual Mean: {np.mean(residuals):.4f}")
        print(f"  Residual Std: {np.std(residuals):.4f}")
        
        return residuals

    def feature_importance_analysis(self):
        """Analyze importance of different features"""
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 35)
        
        # Correlation analysis
        numeric_columns = ['sales', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
                          'rolling_avg_7', 'rolling_avg_30', 'sales_lag1', 'sales_lag7']
        
        correlation_matrix = self.forecaster.df[numeric_columns].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance based on correlation with sales
        sales_correlation = correlation_matrix['sales'].drop('sales').sort_values(ascending=False)
        
        print("\n📈 Feature Correlation with Sales:")
        for feature, corr in sales_correlation.items():
            print(f"  {feature}: {corr:.4f}")
        
        return sales_correlation

    def seasonal_decomposition(self):
        """Perform seasonal decomposition of time series"""
        print("\n📅 SEASONAL DECOMPOSITION ANALYSIS")
        print("=" * 40)
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Use multiplicative decomposition for sales data
        sales_series = self.forecaster.df.set_index('date')['sales']
        
        # Resample to daily frequency if needed
        decomposition = seasonal_decompose(sales_series, model='additive', period=365)
        
        plt.figure(figsize=(15, 12))
        
        # Original series
        plt.subplot(4, 1, 1)
        plt.plot(decomposition.observed)
        plt.title('Original Sales Series')
        plt.ylabel('Sales')
        
        # Trend component
        plt.subplot(4, 1, 2)
        plt.plot(decomposition.trend)
        plt.title('Trend Component')
        plt.ylabel('Trend')
        
        # Seasonal component
        plt.subplot(4, 1, 3)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal Component')
        plt.ylabel('Seasonality')
        
        # Residual component
        plt.subplot(4, 1, 4)
        plt.plot(decomposition.resid)
        plt.title('Residual Component')
        plt.ylabel('Residuals')
        plt.xlabel('Date')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return decomposition

    def run_complete_analysis(self):
        """Run all advanced analysis methods"""
        print("🔬 ADVANCED SALES ANALYSIS")
        print("=" * 30)
        
        # Model comparison
        comparison_results = self.compare_models()
        
        # Residual analysis
        residuals = self.residual_analysis()
        
        # Feature importance
        feature_importance = self.feature_importance_analysis()
        
        # Seasonal decomposition
        decomposition = self.seasonal_decomposition()
        
        print("\n" + "=" * 50)
        print("✅ ADVANCED ANALYSIS COMPLETED!")
        print("=" * 50)
        
        return {
            'comparison': comparison_results,
            'residuals': residuals,
            'feature_importance': feature_importance,
            'decomposition': decomposition
        }