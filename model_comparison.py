"""
Model Comparison Example
Compares LSTM with traditional forecasting methods
"""

import sys
import os
sys.path.append('..')

from sales_forecaster import SalesForecaster
from advanced_analysis import AdvancedSalesAnalysis

def main():
    print("⚡ MODEL COMPARISON DEMO")
    print("=" * 35)
    
    # Initialize and prepare data
    forecaster = SalesForecaster()
    forecaster.generate_realistic_sales_data(days=800)
    
    # Prepare LSTM data
    X_train, X_test, y_train, y_test, split_idx = forecaster.prepare_lstm_data()
    
    # Train LSTM model
    print("\n🎯 Training LSTM model for comparison...")
    forecaster.train_lstm_model(X_train, y_train, X_test, y_test, epochs=50)
    
    # Perform advanced analysis and model comparison
    print("\n🔬 Running model comparison...")
    advanced = AdvancedSalesAnalysis(forecaster)
    
    # Run complete comparison
    results = advanced.run_complete_analysis()
    
    print("\n🏆 Best performing model:")
    best_model = results['comparison'].iloc[0]
    print(f"  {best_model['Method']} with MAE: {best_model['MAE']:.2f}")
    
    print("\n✅ Model comparison demo completed!")

if __name__ == "__main__":
    main()