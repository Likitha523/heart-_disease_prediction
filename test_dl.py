import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("cardio_train.csv", sep=';')
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
df = df[(df['ap_hi'] > df['ap_lo'])] 
df = df[(df['height'] >= 130) & (df['height'] <= 220)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
df['age_years'] = (df['age'] / 365.25)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

def categorize_bp(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80: return 0
    elif 120 <= ap_hi <= 129 and ap_lo < 80: return 1
    elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89: return 2
    elif ap_hi >= 140 or ap_lo >= 90: return 3
    else: return 0
df['bp_risk'] = df.apply(lambda x: categorize_bp(x['ap_hi'], x['ap_lo']), axis=1)

def categorize_age(age):
    if age < 40: return 0
    elif 40 <= age < 50: return 1
    elif 50 <= age < 55: return 2
    elif 55 <= age < 60: return 3
    else: return 4
df['age_group'] = df['age_years'].apply(categorize_age)
df.drop('age', axis=1, inplace=True)

X = df.drop('cardio', axis=1)
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dl_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64),
                             activation='relu', solver='adam', alpha=0.0001,
                             batch_size=128, learning_rate='adaptive', max_iter=150,
                             early_stopping=True, validation_fraction=0.2, random_state=42)
dl_model.fit(X_train_scaled, y_train)
preds = dl_model.predict(X_test_scaled)
print("Deep Learning Accuracy:", accuracy_score(y_test, preds))
