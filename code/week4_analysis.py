import pandas as pd

# Load training dataset
train_data = pd.read_csv("dataset/UNSW_NB15_training-set.csv")

# Load testing dataset
test_data = pd.read_csv("dataset/UNSW_NB15_testing-set.csv")

print("Training Dataset Shape:", train_data.shape)
print("Testing Dataset Shape:", test_data.shape)

print("\nTraining Dataset:")
print(train_data.head())

print("\nTesting Dataset:")
print(test_data.head())

print("\nColumn Names:")
print(train_data.columns.tolist())

print("\nMissing Values:")
print(train_data.isnull().sum())

print("\nDuplicate Rows:")
print("Training:", train_data.duplicated().sum())
print("Testing:", test_data.duplicated().sum())

print("\nTarget Distribution:")
print(train_data["label"].value_counts())

print("\nTarget Percentage:")
print(train_data["label"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Normal vs Attack Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=train_data, x="label")
plt.title("Normal vs Attack Traffic")
plt.xlabel("Label (0 = Normal, 1 = Attack)")
plt.ylabel("Number of Records")
plt.savefig("visualizations/normal_vs_attack.png")
plt.show()


# 2. Protocol Distribution
plt.figure(figsize=(10, 5))
train_data["proto"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Network Protocols")
plt.xlabel("Protocol")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/protocol_distribution.png")
plt.show()


# 3. Attack Category Distribution
plt.figure(figsize=(10, 5))
train_data["attack_cat"].value_counts().plot(kind="bar")
plt.title("Attack Category Distribution")
plt.xlabel("Attack Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/attack_category_distribution.png")
plt.show()

# 4. Correlation Heatmap with Target
numeric_data = train_data.select_dtypes(include="number")

correlation = numeric_data.corr()["label"].abs().sort_values(ascending=False)

top_features = correlation.head(15).index

plt.figure(figsize=(12, 8))
sns.heatmap(
    train_data[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Top Features")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

# Feature and Target Separation

X_train = train_data.drop(columns=["label", "attack_cat", "id"])
y_train = train_data["label"]

X_test = test_data.drop(columns=["label", "attack_cat", "id"])
y_test = test_data["label"]

print("\nFeatures used for training:")
print(X_train.columns.tolist())

print("\nNumber of Features:", X_train.shape[1])

print("\nCategorical Features:")
print(X_train.select_dtypes(include="object").columns.tolist())

print("\nNumerical Features:")
print(X_train.select_dtypes(include="number").columns.tolist())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_features = ["proto", "service", "state"]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("\nPreprocessing and model pipeline created successfully.")

from sklearn.model_selection import cross_val_score

print("\nStarting 5-Fold Cross-Validation...")

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

print("\nCross-Validation F1 Scores:")
print(cv_scores)

print("\nMean Cross-Validation F1 Score:", cv_scores.mean())

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Train the final model
print("\nTraining final Random Forest model...")

pipeline.fit(X_train, y_train)

print("Final model training completed.")

# Predict on testing dataset
y_pred = pipeline.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== TEST SET RESULTS =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix Visualization

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Random Forest")

plt.tight_layout()

plt.savefig("visualizations/confusion_matrix.png")

plt.close()

print("\nConfusion matrix saved successfully.")

# Feature Importance

print("\nCalculating Feature Importance...")

# Get feature names after preprocessing
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# Get importance values from Random Forest
importances = pipeline.named_steps["model"].feature_importances_

# Create a Series
feature_importance = pd.Series(
    importances,
    index=feature_names
).sort_values(ascending=False)

# Display top 15 features
print("\nTop 15 Important Features:")
print(feature_importance.head(15))

# Plot top 15 features
plt.figure(figsize=(10, 6))

feature_importance.head(15).sort_values().plot(kind="barh")

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig("visualizations/feature_importance.png")

plt.close()

print("\nFeature importance saved successfully.")

# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print("\n================ OUTLIER ANALYSIS ================")

outlier_features = [
    "dur",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sttl",
    "dttl",
    "ct_state_ttl"
]

outlier_results = []

for feature in outlier_features:

    Q1 = train_data[feature].quantile(0.25)
    Q3 = train_data[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train_data[
        (train_data[feature] < lower_bound) |
        (train_data[feature] > upper_bound)
    ]

    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(train_data)) * 100

    outlier_results.append([
        feature,
        Q1,
        Q3,
        lower_bound,
        upper_bound,
        outlier_count,
        outlier_percentage
    ])


outlier_df = pd.DataFrame(
    outlier_results,
    columns=[
        "Feature",
        "Q1",
        "Q3",
        "Lower Bound",
        "Upper Bound",
        "Outlier Count",
        "Outlier Percentage"
    ]
)

print("\nOutlier Summary:")
print(outlier_df.to_string(index=False))

outlier_df.to_csv(
    "visualizations/outlier_analysis.csv",
    index=False
)

print("\nOutlier analysis saved successfully.")


# ============================================================
# OUTLIER BOXPLOT
# ============================================================

plt.figure(figsize=(12, 7))

train_data[outlier_features].boxplot()

plt.title("Boxplot of Selected Network Traffic Features")
plt.ylabel("Feature Value")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("visualizations/outlier_boxplots.png")

plt.close()

print("\nOutlier boxplots saved successfully.")

print("\n===================================================")
print("ALL REMAINING ANALYSIS COMPLETED SUCCESSFULLY!")
print("===================================================")