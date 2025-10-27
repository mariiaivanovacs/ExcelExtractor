import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# 1) Load & basic cleaning
# -------------------------
df = pd.read_csv("experiment/only_digits.csv")


print("Before deduplication:", df.shape)
df = df.drop_duplicates()
print("After deduplication:", df.shape)

# Drop columns you don't want as features

X = df.drop(["character", "variant"], axis=1)
X = X[["left_sum","right_sum","top_sum","bottom_sum","avg_1","avg_2",'avg_3',"avg_4",'avg_5',"avg_6","avg_7","avg_8"]]

# X = X[["projection_correlation",'q3_energy_ratio',"q4_energy_ratio","row_entropy","col_entropy","projection_entropy_ratio", "hog_mean"]]
# X = X.drop(["hog_std", "hog_max", "hog_energy", "hu_moment_3", "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7", "projection_variance", "l1_normalized_gradient", "vertical_intensity_variance", "row_entropy", "col_entropy"], axis=1)
# Remove constant columns (you already had this)
# X = X.loc[:, X.std() > 1e-6]

# If there are any non-numeric columns, convert or expand them (one-hot)
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Found non-numeric columns, converting with one-hot:", non_numeric)
    X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

# Check for NaNs in features or target
if X.isnull().any().any():
    print("Warning: NaNs found in X — consider imputing or dropping rows. Example counts:")
    print(X.isnull().sum().loc[lambda s: s>0].head())
    # Optionally drop rows with NaN (or better: impute)
    # X = X.dropna()
    
y = df["character"].astype(str)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

if pd.isna(y_encoded).any():
    raise ValueError("NaNs found in label vector after encoding")

# -------------------------
# 2) Single split (train/test)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)


# dup_rows = X[X.duplicated()]
# print("Duplicate feature rows:", len(dup_rows))
# dup_indices = X[X.duplicated(keep=False)].index
# print(df.loc[dup_indices, ["character"] + list(X.columns)[:5]])  # show first 5 features for readability

# # dup_groups = df.groupby(["mass", "first_moment", "second_moment", "magnitude", "canny_edge_density"])["character"].nunique()
# conflicting = dup_groups[dup_groups > 1]
# print("Conflicting duplicate feature rows (same X, different y):", len(conflicting))

overlap = pd.merge(X_train, X_test, how='inner')
print("Overlapping samples:", len(overlap))


print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Ensure label vectors are 1-D numpy arrays (XGBoost can be picky)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Optional quick checks
assert X_train.shape[0] == y_train.shape[0], "Mismatch train samples"
assert X_test.shape[0] == y_test.shape[0], "Mismatch test samples"

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(12,10))
# sns.heatmap(X.corr(), cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()
from sklearn.svm import SVC

# -------------------------
# 3) Models & pipelines
# -------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
models = {
    # Pipelines ensure scaling is applied inside CV folds properly (prevent leakage).
    # "KNN": make_pipeline(scaler, KNeighborsClassifier(n_neighbors=5)),
    # "LogReg": make_pipeline(scaler, LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")),
    # For tree-based models scaling isn't required but it's harmless to include
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    # "XGBoost": XGBClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     use_label_encoder=False,   # important to silence deprecation warning
    #     eval_metric="mlogloss",
    #     random_state=42,
    #     verbosity=0
    # ), 
    # "SVM": make_pipeline(
    #     StandardScaler(),
    #     SVC(
    #         kernel="rbf",       # 'rbf' usually best for nonlinear problems
    #         C=1.0,              # regularization parameter
    #         gamma="scale",      # auto scaling of kernel coefficient
    #         probability=True,   # optional: enables predict_proba()
    #         random_state=42
    #     )
    # )
}

# -------------------------
# 4) Cross-validate on TRAIN set and then evaluate on TEST set
# -------------------------
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\n=== {name} ===")
    # cross_val_score on the **training** set (no leakage because we use pipelines
    # for those that need scaling). For models that are not pipelines, cross_val_score
    # will still work (scikit-learn will clone and use raw X_train).
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1
        )
        print(f"CV (5-fold stratified) accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print("Error during cross_val_score:", e)
        # continue to try to fit below

    # Fit final model on the full training set
    # If model is a pipeline we pass X_train directly; if not, X_train is fine too.
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error fitting {name}: {e}")
        raise

    # Predict and evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("Train acc:", model.score(X_train, y_train))
    print("Test acc:", model.score(X_test, y_test))
    
    overlap = pd.merge(X_train, X_test, how='inner')
    print(f"Overlapping samples between train/test: {len(overlap)}")
    import numpy as np
    corrs = [np.corrcoef(X[col], y_encoded)[0,1] for col in X.columns]
    pd.Series(corrs, index=X.columns).sort_values(ascending=False).head(10)
    
    import seaborn as sns
    sns.pairplot(df.sample(1000), vars=X.columns[:4], hue="character")



    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    model_path = f"models/{name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")
    
    
    

# -------------------------
# 5) Optional: save label encoder (to recover labels)
# -------------------------
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved label encoder.")
