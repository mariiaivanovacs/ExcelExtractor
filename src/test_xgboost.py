import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# 1️⃣ Load the real-world data
df = pd.read_csv("experiment/test_features.csv")

df = df.drop(["character", "variant" ], axis=1, errors="ignore")

# ⚠️ Make sure to apply the *same preprocessing* you used before training
# For example: dropping columns, selecting same features
X_real = df.drop(["character", "variant"], axis=1, errors="ignore")
X_real = X_real[["left_sum","right_sum","top_sum","bottom_sum","avg_1","avg_2",'avg_3',"avg_4",'avg_5',"avg_6","avg_7","avg_8"]]

# X_real = X_real[[ "center_x", "center_y", 'hu_moment_2', 'hog_energy', 'avg_contour_solidity', 'hog_std', 'compactness', 'first_moment',"projection_correlation" ,"energy_ratio","q1_energy_ratio",'q2_energy_ratio','q3_energy_ratio',"q4_energy_ratio","quadrant_energy_variance","row_variance_ratio","row_entropy","col_entropy","projection_entropy_ratio"]]
# X_real = X_real[["projection_correlation",'q3_energy_ratio',"q4_energy_ratio","row_entropy","col_entropy","projection_entropy_ratio", "hog_mean"]]
# X_real = X_real.drop(["hog_std", "hog_max", "hog_energy", "hu_moment_3", "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7", "projection_variance", "l1_normalized_gradient", "vertical_intensity_variance", "row_entropy", "col_entropy"], axis=1)

# X_real = X_real[['hu_moment_2', 'hog_energy', 'avg_contour_solidity', 'hog_std', 'compactness', 'first_moment', 'hu_moment_1', 'hu_moment_6', 'hog_max', 'hu_moment_5', 'hu_moment_7', 'magnitude', 'hu_moment_4', 'hu_moment_3', 'canny_edge_density', 'projection_variance', 'vertical_intensity_variance', 'edge_direction_ratio', 'mass', 'second_moment']]
# X_real = X_real.loc[:, X_real.std() > 1e-6]

# 2️⃣ Load the saved model
model = joblib.load("models/RandomForest.pkl")  # adjust the path if needed
# model = joblib.load("models/LogReg.pkl")  # adjust the path if needed
# model = joblib.load("models/KNN.pkl")  # adjust the path if needed
# 3️⃣ If you encoded labels before, load the encoder too (optional)
# model = joblib.load("models/SVM.pkl")  # adjust the path if needed
try:
    le = joblib.load("models/label_encoder.pkl")
except FileNotFoundError:
    le = None  # if you only need raw predictions

# laod scaler
# scaler = joblib.load("models/scaler.pkl")
# X_real = scaler.transform(X_real)

# 4️⃣ Make predictions
y_pred = model.predict(X_real)



# 5️⃣ If label encoder is available, decode back to labels
if le is not None:
    print("Label encoder found, decoding predictions...")
    y_pred_labels = le.inverse_transform(y_pred)
else:
    y_pred_labels = y_pred

# 6️⃣ (Optional) Add predictions to DataFrame and save
df["predicted_character"] = y_pred_labels
df.to_csv("experiment/real_features_with_predictions.csv", index=False)

print("✅ Predictions saved to 'experiment/real_features_with_predictions.csv'")

# 7️⃣ (Optional) If you have true labels in the CSV, evaluate
if "character" in df.columns:
    y_true = df["character"].astype(str)
    print("\nAccuracy:", accuracy_score(y_true, y_pred_labels))
    print(classification_report(y_true, y_pred_labels))
