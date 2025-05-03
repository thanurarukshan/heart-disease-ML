import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values (you can do more sophisticated imputation too)
    df = df.dropna()

    # Separate features and target
    X = df.drop("Heart Disease Status", axis=1)
    y = df["Heart Disease Status"]

    # Encode target (Yes/No → 1/0)
    y = y.map({"Yes": 1, "No": 0})

    # Identify categorical columns
    cat_cols = X.select_dtypes(include="object").columns

    # Apply Label Encoding to each categorical column
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # Save for future inverse transform if needed

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
