# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# # import RandomForest
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import numpy as np
# import joblib

# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("experiment/only_2_features.csv",)
# X = df.drop(["character", "variant", "center_x", "center_y"], axis=1)

# # X = X.loc[:, X.std() > 1e-6]
# print(f"Columns: {X.columns}")
# y = df["character"].astype(str)

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)


# # from sklearn.preprocessing import MinMaxScaler
# # scaler = MinMaxScaler()
# # X_scaled = scaler.fit_transform(X)

# # # df_real_data = pd.read_csv("results/features_output.csv")
# # # df_output = df_real_data[["sample_id"]]

# # # df_real_data = df_real_data.drop(["sample_id"], axis=1)
# # # df_real_data = df_real_data[["vertical_intensity_variance", "compactness", "average_peak_width", "average_peak_width", "intensity_fluctuation_ratio", "frequency_white"]]

# # # from sklearn.preprocessing import StandardScaler
# # # X_scaled = StandardScaler().fit_transform(X)
# # # print(X_scaled[:10])

# # from sklearn.model_selection import train_test_split

# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_scaled, y, 
# #     test_size=0.2, 
# #     stratify=y, 
# #     random_state=42  # optional, for reproducibility
# # )
# # # X_real = StandardScaler().fit_transform(df_real_data)

# # models = {
# #     "LogReg": LogisticRegression(),
# #     # "SVM": SVC(kernel='rbf'),
# #     # "KNN": KNeighborsClassifier(n_neighbors=5),
# #     # "RandomForest": RandomForestClassifier(n_estimators=100)
# # }

# # # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# # # axes = axes.flatten()  # to iterate easily
# # # count = 0
# # for name, model in models.items():
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #     # try to test on real data 
    
# #     # print(name)
# #     # disp = ConfusionMatrixDisplay.from_predictions(
# #     #     y_test, y_pred, ax=axes[count], cmap='Blues', colorbar=False
# #     # )
# #     # axes[count].set_title(name)
# #     # count += 1
# #     # if count == 3:
# #     #     y_pred_real = model.predict(X_real)

# #     #     if y_pred_real.ndim > 1:
# #     #         y_pred_labels = np.argmax(y_pred_real, axis=1)
# #     #     else:
# #     #         y_pred_labels = y_pred_real  # direct output for binary/class models

# #     #     # Add predictions to the dataframe
# #     #     df_output["predicted_label"] = y_pred_labels

# #     #     # Save only the columns you want (optional)
# #     #     df_output[["sample_id", "predicted_label"]].to_csv("results/predictions_with_id.csv", index=False)

# #     print(classification_report(y_test, y_pred))
# #     scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# #     print(f"Cross-validation scores: {scores}")
# #     print(f"Mean cross-validation score: {scores.mean()}")
    
    


# # plt.tight_layout()
# # plt.savefig("results/confusion_matrix.png")
# # plt.close()

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # Split first
# # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# X_train, X_test, y_encoded_train, y_encoded_test = train_test_split(
#     X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
# )
# print("X_train columns:", X_train.shape[1])
# print("X_test columns:", X_test.shape[1])
# # Fit only on training data, then transform both
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# df_real = pd.read_csv("experiment/real_features.csv")
# list_of_columns = ['first_moment', 'second_moment', 'magnitude', 'canny_edge_density',
#        'sobel_edge_density', 'sobel_mean_magnitude', 'sobel_std_magnitude',
#        'edge_direction_ratio', 'hog_std', 'hog_max', 'hog_energy', 'symmetry',
#        'aspect_ratio', 'projection_variance', 'vertical_intensity_variance',
#        'l1_normalized_gradient', 'num_contours', 'avg_contour_solidity',
#        'avg_contour_area', 'compactness', 'cc_count', 'avg_cc_area',
#        'cc_area_std', 'hu_moment_1', 'hu_moment_2']
# # X = df.drop(["character", "variant", "center_x", "center_y"], axis=1)
# X_real = df_real.drop(["character", "variant", "center_x", "center_y"], axis=1)

# record_index = 5
# record = X_real.iloc[record_index]  # already has only numeric features

# record_2d = np.array(record).reshape(1, -1)
# record_scaled = scaler.transform(record_2d)
# # record_2d = np.array(record).reshape(1, -1)

# # # Scale using the saved scaler
# # record_scaled = scaler.transform(record_2d)

# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import cross_val_score

# # clf = LogisticRegression(max_iter=1000)  # increase iterations
# # scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
# # print("Cross-validation scores:", scores)

# # # # 4️⃣ Initialize models
# # from xgboost import XGBClassifier
# # # # command to install xgboost: pip install xgboost
# # models = {
# #     # "KNN": KNeighborsClassifier(n_neighbors=5),
# #     # "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
# #     # "RandomForest": RandomForestClassifier(
# #     #     n_estimators=200, max_depth=None, random_state=42
# #     # ), 
# #     "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6)
# # }



# # # 5️⃣ Train and evaluate each model
# # for name, model in models.items():
# #     print(f"\n=== {name} ===")
   
# #     if name == "XGBoost":
# #         model.fit(X_train_scaled, y_encoded_train)
# #     # else: 
# #     #     model.fit(X_train_scaled, y_train)
# #     y_pred = model.predict(X_test_scaled)
# #     acc = accuracy_score(y_encoded_test, y_pred)
# #     print(f"Accuracy: {acc:.4f}")
# #     print(classification_report(y_encoded_test, y_pred))
# #     import xgboost as xgb
    
    
    
    
# #     # df_real = pd.read_csv("experiment/"



# #     # Save model (XGBoost native format)
# #     # model.save_model('models/xgboost_model.json')
    
# #     # joblib.dump(scaler, "models/scaler.pkl")
# #     # joblib.dump(le, "models/label_encoder.pkl")

# #     # Optional: cross-validation
# #     scores = cross_val_score(model, X_train_scaled, y_encoded_train, cv=5)
# #     print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("experiment/only_2_features.csv")

X = df.drop(["character", "variant", "center_x", "center_y"], axis=1)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

y = df["character"].astype(str)
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train model (Logistic Regression for example) ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))



import pandas as pd
import numpy as np

# === Load the real image features ===
df_real = pd.read_csv("experiment/real_features.csv")

# Drop the non-feature columns (keep only numeric ones)
X_real = df_real.drop(["character", "variant", "center_x", "center_y"], axis=1)
X_real = X_real.replace([np.inf, -np.inf], np.nan)
X_real= X_real.fillna(0)
# Pick the specific record by index
record_index = 11
record = X_real.iloc[record_index]

print(f"\n🧩 Checking real image at index {record_index}")
print(f"Character in file: {df_real.loc[record_index, 'character']}")

# Reshape to 2D (model expects 2D input)
record_2d = np.array(record).reshape(1, -1)

# === Apply same preprocessing (scaling) ===
record_scaled = scaler.transform(record_2d)

# === Predict ===
pred_encoded = clf.predict(record_scaled)
pred_label = le.inverse_transform(pred_encoded)

print(f"\n🔹 Predicted encoded class: {pred_encoded[0]}")
print(f"🔹 Predicted label: {pred_label[0]}")

# === Optional: show probabilities if supported ===
if hasattr(clf, "predict_proba"):
    probs = clf.predict_proba(record_scaled)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_labels = le.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]
    print("\n🔹 Top 3 most probable classes:")
    for lbl, p in zip(top3_labels, top3_probs):
        print(f"{lbl}: {p:.4f}")



import pandas as pd
importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': clf.coef_[0]
}).sort_values(by='coefficient', ascending=False)
print(importance)
# Cross-validation
# scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
# print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# === Save everything ===
# joblib.dump(clf, "models/logreg_model.pkl")
# joblib.dump(scaler, "models/scaler.pkl")
# joblib.dump(le, "models/label_encoder.pkl")
print("✅ Model, scaler, and label encoder saved!")
