import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import random forest
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("results/features_second_2.csv",)
# df[["label", "dark_frac", "aspect_ratio", "compactness", "col_peaks", "vertical_intensity_variance", "intensity_fluctuation_ratio", "frequency_white", "average_peak_width"]]
# df[["label", "compactness", "vert_strokes_count", "cc_count", "col_peaks", "vertical_intensity_variance", "intensity_fluctuation_ratio", "frequency_white"]]
# df[["label","aspect_ratio","vert_strokes_count","compactness","cc_count", "col_peaks", "vertical_intensity_variance", "frequency_white", "av"]]
# df[["label", "vertical_intensity_variance", "compactness", "average_peak_width", "average_peak_width", "intensity_fluctuation_ratio", "frequency_white"]]
X = df.drop(["character", "variant", "center_x", "center_y"], axis=1)

X = X.loc[:, X.std() > 1e-6]
y = df["character"].astype(str)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
from sklearn.preprocessing import StandardScaler
# Fit only on training data, then transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
feat_names = X.columns

plt.barh(feat_names, importances)
plt.xlabel("Importance")
plt.show()
