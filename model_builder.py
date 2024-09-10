from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def identify_problem_type(df, target_column):
    """Identify if the problem is Classification or Regression."""
    if df[target_column].dtype == 'object' or len(df[target_column].unique()) < 20:
        return "Classification"
    else:
        return "Regression"

def evaluate_models(X_train, X_test, y_train, y_test, problem_type):
    """Train and evaluate models based on the problem type."""
    results = {}

    if problem_type == "Classification":
        models = {
            "Random Forest Classifier": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy
    else:
        models = {
            "Random Forest Regressor": RandomForestRegressor(),
            "Linear Regression": LinearRegression()
        }
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results[model_name] = mse

    return results
