import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@pytest.fixture
def load_data():
    """Fixture to load the California Housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return X, y


def test_data_shape(load_data):
    """Test to ensure the dataset has the expected shape."""
    X, y = load_data
    assert X.shape[0] == y.shape[0], "Feature and target row counts must match"
    assert X.shape[1] == 8, "Feature count must be 8 (as in California Housing)"


def test_train_test_split(load_data):
    """Test to ensure that the train-test split is done correctly."""
    X, y = load_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if the split ratio is correct
    assert len(X_train) == 16512, "Training set size should be 16512 (80% of 20640)"
    assert len(X_test) == 4128, "Test set size should be 4128 (20% of 20640)"


def test_model_training(load_data):
    """Test to ensure that the LinearRegression model trains without errors."""
    X, y = load_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Check if coefficients are not None
    assert (
        model.coef_ is not None
    ), "Model coefficients should not be None after training"
    assert (
        model.intercept_ is not None
    ), "Model intercept should not be None after training"


def test_model_prediction(load_data):
    """Test the model's prediction on the test set."""
    X, y = load_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Ensure the prediction length matches the test set length
    assert len(y_pred) == len(
        y_test
    ), "Prediction length must match the test set length"

    # Check if predictions are not NaN or infinite
    assert np.all(np.isfinite(y_pred)), "Predictions should be finite numbers"


def test_model_evaluation(load_data):
    """Test the model evaluation using mean squared error."""
    X, y = load_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Assert that MSE is a reasonable positive number
    assert mse > 0, "Mean Squared Error should be greater than 0"
    assert (
        mse < 1
    ), "Mean Squared Error should be less than 1 for a simple linear regression model"
