import pandas as pd
import numpy as np

df = pd.read_csv("chess_data.csv")

#Başlamamış satranç maçlarını status sütunundan çıkarıyoruz.
df = df[df['status'] != 'noStart']

#Kullanacağımız sütunlardaki NaN değerleri atıyoruz.
df = df.dropna(subset=['white_rating', 'black_rating', 'winner', 'pgn', 'time_control'])

#Kullanılacak veri sayısını belirleme
df = df.sample(n=25000, random_state=42)

print(df.head())

#pgn sütunu satranç maçındaki hamleleri string halinde tutanbir sütun. Bu sütundan sadece ilk hamleyi (açılış hamlesi) çekiyoruz.
def simplify_opening(pgn):
    moves = pgn.strip().split()
    return moves[0] if len(moves) > 0 else "Unknown"

df["Opening"] = df["pgn"].apply(simplify_opening)

print(df[["Opening"]].head())


#Features ve target seçimi
features = ["white_rating", "black_rating", "time_control", "Opening"]
target = "winner"

X = df[features].copy()
y = df[target].copy()


#Eksik veri temizliği
X = X.dropna()
y = y.loc[X.index]


#Label Encoder kullaanarak kategorik verilerimizi numeric verilere çeviriyoruz.
from sklearn.preprocessing import LabelEncoder

#time_control encode
le_time = LabelEncoder()
X["time_control"] = le_time.fit_transform(X["time_control"].astype(str))

#Opening encode
le_opening = LabelEncoder()
X["Opening"] = le_opening.fit_transform(X["Opening"].astype(str))

#winner encode
le_y = LabelEncoder()
y = le_y.fit_transform(y)

print("Encoded classes:", le_y.classes_)


#Train-test split ile veriyi %80'nini eğitim %20'sini de test olmak üzere parçalıyoruz.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Logistic Regression modeli
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#Performans ölçümü
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_y.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#ROC AUC (Binary ise) — White vs Black için
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#Eğer 2 sınıf varsa ROC çizer
if len(le_y.classes_) == 2:
    y_scores = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()

    auc = roc_auc_score(y_test, y_scores[:,1])
    print("AUC Score:", auc)
else:
    print("ROC yalnızca 2 sınıflı veride hesaplanır.")

