import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("Loading student application data...")
df = pd.read_csv('student_applications.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Admission rate: {df['admitted'].mean():.2%}")

# Feature engineering
print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Create predictive features
df['sat_above_college_75'] = (df['student_sat_math'] >= df['college_sat_math_75']).astype(int)
df['sat_above_college_25'] = (df['student_sat_math'] >= df['college_sat_math_25']).astype(int)
df['gpa_high'] = (df['student_gpa'] >= 3.7).astype(int)
df['sat_college_diff'] = df['student_sat_math'] - df['college_sat_math_25']
df['sat_percentile_position'] = (df['student_sat_math'] - df['college_sat_math_25']) / (
            df['college_sat_math_75'] - df['college_sat_math_25'])
df['academic_strength'] = df['student_gpa'] * df['student_sat_total'] / 1000  # Combined academic metric

print("\nFeatures created:")
print("- sat_above_college_75: SAT above 75th percentile")
print("- sat_above_college_25: SAT above 25th percentile")
print("- gpa_high: GPA >= 3.7")
print("- sat_college_diff: SAT difference from 25th percentile")
print("- sat_percentile_position: Where student falls in college's SAT range")
print("- academic_strength: Combined GPA and SAT metric")

# Select features for model
feature_columns = [
    'student_gpa',
    'student_sat_math',
    'student_sat_reading',
    'student_sat_total',
    'has_research_experience',
    'leadership_score',
    'num_ap_courses',
    'college_admit_rate',
    'sat_above_college_75',
    'sat_above_college_25',
    'gpa_high',
    'sat_college_diff',
    'sat_percentile_position',
    'academic_strength'
]

X = df[feature_columns]
y = df['admitted']

print(f"\nTotal features: {len(feature_columns)}")

# Train/test split
print("\n" + "=" * 50)
print("TRAIN/TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} applications")
print(f"Test set: {len(X_test)} applications")

# Train multiple models
print("\n" + "=" * 50)
print("TRAINING MODELS")
print("=" * 50)

models = {}
accuracies = {}

# Model 1: Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
models['Logistic Regression'] = lr_model
accuracies['Logistic Regression'] = lr_accuracy
print(f"   Accuracy: {lr_accuracy:.2%}")

# Model 2: Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf_model
accuracies['Random Forest'] = rf_accuracy
print(f"   Accuracy: {rf_accuracy:.2%}")

# Model 3: Gradient Boosting
print("\n3. Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
models['Gradient Boosting'] = gb_model
accuracies['Gradient Boosting'] = gb_accuracy
print(f"   Accuracy: {gb_accuracy:.2%}")

# Compare models
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:25s} {acc:.2%}")

# Select best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")

# Detailed evaluation
print("\n" + "=" * 50)
print(f"DETAILED EVALUATION - {best_model_name}")
print("=" * 50)

best_pred = best_model.predict(X_test)
best_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Rejected', 'Admitted']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)
print("\n[True Negatives  False Positives]")
print("[False Negatives True Positives]")

try:
    auc_score = roc_auc_score(y_test, best_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
except:
    pass

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop features for predicting admission:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:35s} {row['importance']:.4f}")

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10))
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved feature importance chart: 'feature_importance.png'")

# Save the model
import pickle

with open('admission_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n" + "=" * 50)
print("✅ Model saved as 'admission_model.pkl'")
print("=" * 50)
print(f"\nFinal Model: {best_model_name}")
print(f"Accuracy: {accuracies[best_model_name]:.2%}")
print("\nNext: Run ed_recommender.py to get ED recommendations!")