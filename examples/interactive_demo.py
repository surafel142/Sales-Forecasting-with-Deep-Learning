"""
Interactive Sales Forecasting Demo
Provides a user-friendly interface for sales forecasting
"""

import sys
import os
sys.path.append('..')

from sales_forecaster import SalesForecaster

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
            print("4. Explore sales patterns")
            print("5. Retrain model")
            print("6. Exit")

            choice = input("Enter your choice (1-6): ")

            if choice == '1':
                days = 7
            elif choice == '2':
                days = 30
            elif choice == '3':
                days = int(input("Enter number of days to forecast: "))
            elif choice == '4':
                forecaster.visualize_sales_patterns()
                continue
            elif choice == '5':
                print("\n🔄 Retraining model...")
                X_train, X_test, y_train, y_test, split_idx = forecaster.prepare_lstm_data()
                forecaster.train_lstm_model(X_train, y_train, X_test, y_test, epochs=50)
                continue
            elif choice == '6':
                break
            else:
                print("❌ Invalid choice!")
                continue

            if choice in ['1', '2', '3']:
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

def main():
    print("🎮 INTERACTIVE SALES FORECASTING DEMO")
    print("=" * 45)

    # Initialize forecaster
    forecaster = SalesForecaster()

    # Generate data
    print("\n📊 Generating sales data...")
    forecaster.generate_realistic_sales_data(days=600)

    # Prepare and train initial model
    print("\n🤖 Preparing initial model...")
    X_train, X_test, y_train, y_test, split_idx = forecaster.prepare_lstm_data()
    forecaster.train_lstm_model(X_train, y_train, X_test, y_test, epochs=30)

    # Start interactive session
    interactive_forecasting(forecaster)

    print("\n👋 Thank you for using the interactive forecasting demo!")

if __name__ == "__main__":
    main()
