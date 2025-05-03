import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Label Encoding
    label_cols = ['Gender', 'Smoking', 'Diabetes', 'Heart Disease Status', 
                  'High Blood Pressure', 'Low HDL Cholesterol', 
                  'High LDL Cholesterol', 'Family Heart Disease']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Exercise Habits', 'Stress Level'], drop_first=True)

    # Scale numerical features
    num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
                'Triglyceride Level', 'Fasting Blood Sugar', 
                'CRP Level', 'Homocysteine Level', 'Sleep Hours']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Feature/Target split
    X = df.drop("Heart Disease Status", axis=1)
    y = df["Heart Disease Status"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
