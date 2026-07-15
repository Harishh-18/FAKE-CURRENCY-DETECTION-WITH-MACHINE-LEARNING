import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv('data_banknote_authentication.csv', names=columns)
 
# 2. Split features and target
X = df.drop('class', axis=1)
y = df['class']
 
# 3. Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
 
# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# 5. Define candidate models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
 
# 6. Train and evaluate each model
best_model, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f'{name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    if acc > best_acc:
        best_acc, best_model, best_name = acc, model, name
 
print(f'Best Model: {best_name} with Accuracy: {best_acc:.4f}')
 
# 7. Save the best model and scaler for deployment
joblib.dump(best_model, 'fake_currency_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
 
# 8. Predict on a new note's feature values
def predict_note(variance, skewness, curtosis, entropy):
    features = np.array([[variance, skewness, curtosis, entropy]])
    features_scaled = scaler.transform(features)
    prediction = best_model.predict(features_scaled)[0]
    return 'Fake' if prediction == 1 else 'Genuine'
 
# Example usage
result = predict_note(-2.3, 5.1, -3.2, 0.4)
print('Prediction:', result)
