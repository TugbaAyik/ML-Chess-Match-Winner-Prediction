import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Veri yükleme ve temizleme
df = pd.read_csv("chess_data.csv")
df = df[df['status'] != 'noStart']
df = df.dropna(subset=['white_rating', 'black_rating', 'winner', 'pgn', 'time_control'])
df = df.sample(n=25000, random_state=42)  # Az veri ile test

#Açılış tipi çıkar (ilk 2 hamle)
def get_opening(pgn):
    moves = pgn.strip().split()
    return " ".join(moves[:2]) if len(moves) >= 2 else moves[0] if moves else "Unknown"

df.loc[:, 'Opening'] = df['pgn'].apply(get_opening)

#Girdi ve hedef
features = ['white_rating', 'black_rating', 'time_control', 'Opening']
target = 'winner'

X = df[features].copy()
y = df[target]

#Label Encoding
le_time = LabelEncoder()
X.loc[:, 'time_control'] = le_time.fit_transform(X['time_control'].astype(str))

le_opening = LabelEncoder()
X.loc[:, 'Opening'] = le_opening.fit_transform(X['Opening'])

le_y = LabelEncoder()
y = le_y.fit_transform(y)

#Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Ölçekleme (SVM ve Logistic Regression için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelleri tanımla
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Linear Regression": LinearRegression()
}

#Tahmin ve performans
for name, model in models.items():
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    if name == "Linear Regression":
        y_pred = y_pred.round().astype(int)
        y_pred = y_pred.clip(0, len(le_y.classes_)-1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
