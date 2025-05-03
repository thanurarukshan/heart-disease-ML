from data_preprocessing import preprocess_data
from model_training import train_model, evaluate_model, save_model

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocess_data("heart_disease.csv")

# Train model
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)

# Save model
save_model(model)
