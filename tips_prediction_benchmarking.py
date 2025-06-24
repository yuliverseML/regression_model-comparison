# Import necessary libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# =============== DATA LOADING AND EXPLORATION ===============
def load_and_explore_data():
    """Load the tips dataset and perform initial exploration."""
    # Load the dataset
    df = sns.load_dataset('tips')
    
    # Display basic information about the dataset
    print("Dataset head:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print("\nDataset info:")
    df.info()
    print("\nDescriptive statistics:")
    print(df.describe())
    
    return df

# =============== DATA PREPROCESSING ===============
def preprocess_data(df):
    """Encode categorical variables and prepare data for modeling."""
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'])
    
    # Split into features and target
    X = df_encoded.drop('tip', axis=1)
    y = df_encoded['tip']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X, y, X_scaled, X_train, X_test, y_train, y_test, scaler, df_encoded

# =============== MODEL TRAINING AND EVALUATION ===============
def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_scaled, y):
    """Train multiple regression models and evaluate their performance."""
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        results[name] = {
            'MSE': mse,
            'R2': r2,
            'CV_R2_mean': np.mean(cv_scores),
            'CV_R2_std': np.std(cv_scores)
        }
    
    # Print model results
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return models, results

# =============== VISUALIZATION FUNCTIONS ===============
def plot_predictions(models, results, X_test, y_test):
    """Plot predictions vs actual values for all models."""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models.items(), colors):
        y_pred = model.predict(X_test)
        plt.scatter(
            y_test, y_pred, alpha=0.5, color=color, 
            label=f"{name} (R2: {results[name]['R2']:.2f}, MSE: {results[name]['MSE']:.2f})"
        )
    
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'k--', lw=2, label='Ideal line')
    plt.xlabel('Actual tips', fontsize=12)
    plt.ylabel('Predicted tips', fontsize=12)
    plt.title('Comparison of actual vs predicted tips for different models', fontsize=14)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(models, results):
    """Plot comparison of metrics for all models."""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    for i, metric in enumerate(['R2', 'MSE']):
        values = [results[name][metric] for name in models]
        plt.bar(x + i*width, values, width, label=metric, 
                color='blue' if metric=='R2' else 'red')
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Metric value', fontsize=12)
    plt.title('Comparison of metrics across models', fontsize=14)
    plt.xticks(x + width/2, models.keys(), rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def print_model_comparison_table(results):
    """Print formatted table with model comparison."""
    print("\nModel results:")
    print("-" * 80)
    print(f"{'Model':<20} {'MSE':>10} {'R2':>10} {'CV R2 mean':>12} {'CV R2 std':>10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['MSE']:>10.4f} {metrics['R2']:>10.4f} "
              f"{metrics['CV_R2_mean']:>12.4f} {metrics['CV_R2_std']:>10.4f}")

# =============== FINDING THE BEST MODEL ===============
def find_best_model(models, results, X_test, y_test):
    """Find the best model based on R2 and MSE metrics."""
    # Identify best model (maximizing R2 while minimizing MSE)
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'] - x[1]['MSE'])[0]
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    
    # Plot best model predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Ideal line')
    plt.xlabel('Actual tips', fontsize=12)
    plt.ylabel('Predicted tips', fontsize=12)
    plt.title(f'Actual vs predicted tips ({best_model_name})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print best model metrics
    print(f"\nBest model by our criteria: {best_model_name}")
    print(f"Mean Squared Error: {results[best_model_name]['MSE']:.4f}")
    print(f"R-squared Score: {results[best_model_name]['R2']:.4f}")
    
    return best_model, best_model_name

# =============== MAKE PREDICTIONS ===============
def predict_tip_for_new_bill(best_model, df, X, scaler, best_model_name):
    """Make a prediction for a new bill amount."""
    # Generate a random bill amount
    new_total_bill = np.random.uniform(df['total_bill'].min(), df['total_bill'].max())
    
    # Create a new observation
    new_observation = pd.DataFrame({
        'total_bill': [new_total_bill],
        'size': [df['size'].mean()],
    })
    
    # Add missing columns with appropriate default values
    for col in X.columns:
        if col not in new_observation.columns:
            if col.startswith(('sex_', 'smoker_', 'day_', 'time_')):
                new_observation[col] = 0
            else:
                new_observation[col] = X[col].mean()
    
    # Ensure correct column order
    new_observation = new_observation.reindex(columns=X.columns, fill_value=0)
    
    # Scale the new observation
    new_observation_scaled = scaler.transform(new_observation)
    
    # Make prediction
    predicted_tip = best_model.predict(new_observation_scaled)[0]
    
    print(f"\nPredicted tip for a bill of ${new_total_bill:.2f}: ${predicted_tip:.2f}")
    
    return new_total_bill, predicted_tip

# =============== FEATURE ENGINEERING ===============
def experiment_with_random_feature(X, X_train, X_test, y, y_train, y_test, results, scaler):
    """Experiment with adding a random feature to see its effect on model performance."""
    # Create a random feature
    np.random.seed(42)
    new_feature = np.random.randn(len(X))
    
    # Add the feature to the dataset
    X_with_new_feature = np.column_stack((X, new_feature))
    X_scaled_with_new = scaler.fit_transform(X_with_new_feature)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_scaled_with_new, y, test_size=0.2, random_state=42
    )
    
    # Train a model with the new feature
    model_new = LinearRegression()
    model_new.fit(X_train_new, y_train)
    y_pred_new = model_new.predict(X_test_new)
    
    # Evaluate the model
    mse_new = mean_squared_error(y_test, y_pred_new)
    r2_new = r2_score(y_test, y_pred_new)
    
    # Print results
    print("\nModel with random feature:")
    print(f"Mean Squared Error: {mse_new:.4f}")
    print(f"R-squared Score: {r2_new:.4f}")
    
    # Compare with previous model
    original_mse = results['Linear Regression']['MSE']
    original_r2 = results['Linear Regression']['R2']
    
    print("\nComparison with previous model:")
    print(f"MSE (original model): {original_mse:.4f}")
    print(f"MSE (model with random feature): {mse_new:.4f}")
    print(f"R2 (original model): {original_r2:.4f}")
    print(f"R2 (model with random feature): {r2_new:.4f}")
    
    # Check if adding the feature improved the model
    if r2_new > original_r2 or (r2_new >= original_r2 and mse_new < original_mse):
        print("Adding the random feature improved the model.")
    else:
        print("Adding the random feature did not improve the model or the improvement is negligible.")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': list(X.columns) + ['Random Feature'],
        'Importance': np.abs(model_new.coef_)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance (absolute coefficient value)')
    plt.title('Feature importance in the model with random feature')
    plt.tight_layout()
    plt.show()
    
    # Plot random feature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(new_feature, kde=True)
    plt.title('Distribution of the random feature')
    plt.xlabel('Random feature value')
    plt.ylabel('Density / Frequency')
    plt.show()
    
    return feature_importance, new_feature

def try_polynomial_features(X, y):
    """Experiment with polynomial features."""
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Split data
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # Create and train a polynomial regression model
    poly_model = make_pipeline(StandardScaler(), LinearRegression())
    poly_model.fit(X_train_poly, y_train_poly)
    
    # Evaluate the model
    y_pred_poly = poly_model.predict(X_test_poly)
    mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
    r2_poly = r2_score(y_test_poly, y_pred_poly)
    
    print("\nPolynomial regression:")
    print(f"Mean Squared Error: {mse_poly:.4f}")
    print(f"R-squared Score: {r2_poly:.4f}")
    
    return poly_model, X_train_poly, X_test_poly, y_train_poly, y_test_poly, mse_poly, r2_poly

# =============== TESTING FUNCTIONS ===============
def run_tests(df, X_train, X_test, X_scaled, y, y_train, y_test, best_model, y_pred_best, best_model_name, results):
    """Run tests to ensure code correctness."""
    def test_data_loading():
        assert isinstance(df, pd.DataFrame), "Data not loaded into DataFrame"
        assert 'tip' in df.columns, "Column 'tip' missing from data"
        print("Data loading test passed")

    def test_data_splitting():
        assert X_train.shape[0] + X_test.shape[0] == X_scaled.shape[0], "Incorrect data splitting"
        assert y_train.shape[0] + y_test.shape[0] == y.shape[0], "Incorrect data splitting"
        print("Data splitting test passed")

    def test_model_training():
        assert hasattr(best_model, 'predict'), "Model not trained"
        print("Model training test passed")

    def test_predictions():
        assert len(y_pred_best) == len(y_test), "Number of predictions doesn't match test set size"
        print("Predictions test passed")

    def test_model_performance():
        assert 0 <= results[best_model_name]['R2'] <= 1, "R-squared should be between 0 and 1"
        assert results[best_model_name]['MSE'] >= 0, "MSE should be non-negative"
        print("Model performance test passed")

    test_data_loading()
    test_data_splitting()
    test_model_training()
    test_predictions()
    test_model_performance()
    print("All tests passed successfully!")

# =============== MAIN FUNCTION ===============
def main():
    """Main function to run the entire analysis pipeline."""
    try:
        # Data loading and exploration
        df = load_and_explore_data()
        
        # Data preprocessing
        X, y, X_scaled, X_train, X_test, y_train, y_test, scaler, df_encoded = preprocess_data(df)
        
        # Model training and evaluation
        models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test, X_scaled, y)
        
        # Visualizations
        plot_predictions(models, results, X_test, y_test)
        plot_metrics_comparison(models, results)
        print_model_comparison_table(results)
        
        # Find best model
        best_model, best_model_name = find_best_model(models, results, X_test, y_test)
        y_pred_best = best_model.predict(X_test)
        
        # Make predictions
        new_total_bill, predicted_tip = predict_tip_for_new_bill(best_model, df, X, scaler, best_model_name)
        
        # Feature engineering experiments
        feature_importance, new_feature = experiment_with_random_feature(
            X, X_train, X_test, y, y_train, y_test, results, scaler
        )
        
        # Try polynomial features
        poly_results = try_polynomial_features(X, y)
        poly_model, X_train_poly, X_test_poly, y_train_poly, y_test_poly, mse_poly, r2_poly = poly_results
        
        # Compare polynomial model with best linear model
        print("\nComparison with best linear model:")
        print(f"MSE (best linear model): {results[best_model_name]['MSE']:.4f}")
        print(f"MSE (polynomial model): {mse_poly:.4f}")
        print(f"R2 (best linear model): {results[best_model_name]['R2']:.4f}")
        print(f"R2 (polynomial model): {r2_poly:.4f}")
        
        # Run tests
        run_tests(
            df, X_train, X_test, X_scaled, y, y_train, y_test, 
            best_model, y_pred_best, best_model_name, results
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()




