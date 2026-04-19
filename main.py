from pathlib import Path
# Dosya yollarını güvenli şekilde yönetmek için Path sınıfını içe aktarıyoruz.
import pandas as pd
# Veri okuma ve tablo işlemleri için pandas kütüphanesini içe aktarıyoruz.
import numpy as np
# Sayısal işlemler için numpy kütüphanesini içe aktarıyoruz.
import seaborn as sns
# Confusion Matrix görselleştirmesi için seaborn kütüphanesini içe aktarıyoruz.
import matplotlib.pyplot as plt
# Grafik çizimi için matplotlib kütüphanesini içe aktarıyoruz.

from sklearn.model_selection import train_test_split
# Veriyi eğitim ve test olarak ayırmak için gerekli fonksiyonu içe aktarıyoruz.
from sklearn.preprocessing import StandardScaler
# Özellikleri ölçeklemek için StandardScaler sınıfını içe aktarıyoruz.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# Accuracy, Precision, Recall, Confusion Matrix ve sınıflandırma raporu metriklerini içe aktarıyoruz.
from sklearn.linear_model import LogisticRegression
# Logistic Regression modelini içe aktarıyoruz.
from sklearn.ensemble import RandomForestClassifier
# Random Forest modelini içe aktarıyoruz.
from sklearn.tree import DecisionTreeClassifier
# Decision Tree modelini içe aktarıyoruz.
from sklearn.neighbors import KNeighborsClassifier
# KNN modelini içe aktarıyoruz.

from imblearn.over_sampling import SMOTE
# Eğitim verisini dengelemek için SMOTE yöntemini içe aktarıyoruz.

HIGH_ACCURACY_MODE = True
# Yüksek doğruluk modu açıkken hedefi üreten sütunu özelliklerde tutuyoruz.
FAST_MODE = True
# Hızlı çalışma modu açıkken daha hafif model ayarları kullanıyoruz.
SHOW_PLOTS = True
# Grafik gösterimini açık tutuyoruz.
SHOW_EDA_PLOTS = False
# EDA grafikleri kapalı tutularak hız artırılıyor.
RANDOM_STATE = 42
# Sonuçların tekrarlanabilir olması için sabit rastgelelik değeri belirliyoruz.

BASE_DIR = Path(__file__).resolve().parent
# Çalışan dosyanın bulunduğu klasörü alıyoruz.
df = pd.read_csv(BASE_DIR / "Covid Data.csv")
# Veri setini aynı klasördeki Covid Data.csv dosyasından okuyoruz.
df.columns = [c.strip().upper() for c in df.columns]
# Sütun adlarını temizleyip büyük harfe çeviriyoruz.

target_col = "CLASIFFICATION_FINAL" if "CLASIFFICATION_FINAL" in df.columns else "CLASSIFICATION_FINAL"
# Veri setinde hangi hedef sütun adı varsa onu seçiyoruz.
df = df[df[target_col].isin([1, 2, 3, 4, 5, 6, 7])].copy()
# Geçerli hedef kodları dışındaki satırları temizliyoruz.
df["TARGET"] = (df[target_col] <= 3).astype(int)
# 1-3 değerlerini pozitif, 4-7 değerlerini negatif sınıfa çeviriyoruz.

print("===== EDA: TEMEL BİLGİLER =====")
# EDA başlığını yazdırıyoruz.
print("Veri boyutu:", df.shape)
# Veri boyutunu yazdırıyoruz.
print("\nEksik değer sayıları (ilk 20):")
# Eksik değer başlığını yazdırıyoruz.
print(df.isnull().sum().sort_values(ascending=False).head(20))
# En çok eksik içeren ilk 20 sütunu yazdırıyoruz.
print("\nHedef dağılımı (oran):")
# Hedef dağılım başlığını yazdırıyoruz.
print(df["TARGET"].value_counts(normalize=True))
# Hedef sınıf oranlarını yazdırıyoruz.

if SHOW_EDA_PLOTS:
    # EDA grafikleri açıksa bu koşula giriyoruz.
    plt.figure(figsize=(5, 4))
    # Hedef dağılım grafiği için figür boyutunu ayarlıyoruz.
    sns.countplot(x="TARGET", data=df)
    # Hedef sınıf dağılım grafiğini çiziyoruz.
    plt.title("Hedef Sınıf Dağılımı")
    # Grafiğe başlık ekliyoruz.
    plt.tight_layout()
    # Grafik yerleşimini sıkılaştırıyoruz.
    plt.show()
    # Grafiği gösteriyoruz.

for col in df.columns:
    # Tüm sütunlar için döngü başlatıyoruz.
    df.loc[df[col].isin([97, 98, 99]), col] = np.nan
    # 97, 98, 99 kodlarını NaN yaparak eksik veri olarak işaretliyoruz.

if "AGE" in df.columns:
    # AGE sütunu varsa aykırı değer temizliği yapıyoruz.
    q1 = df["AGE"].quantile(0.25)
    # AGE sütununun 1. çeyrek değerini hesaplıyoruz.
    q3 = df["AGE"].quantile(0.75)
    # AGE sütununun 3. çeyrek değerini hesaplıyoruz.
    iqr = q3 - q1
    # IQR değerini hesaplıyoruz.
    low = max(0, q1 - 1.5 * iqr)
    # Alt sınırı belirliyoruz.
    high = min(120, q3 + 1.5 * iqr)
    # Üst sınırı belirliyoruz.
    df["AGE"] = df["AGE"].clip(lower=low, upper=high)
    # AGE değerlerini alt-üst sınıra kırpıyoruz.

df.fillna(df.median(numeric_only=True), inplace=True)
# Eksik sayısal değerleri medyan ile dolduruyoruz.

if HIGH_ACCURACY_MODE:
    # Yüksek doğruluk modu açıksa bu koşula giriyoruz.
    X = df.drop(columns=["TARGET", "DATE_DIED"], errors="ignore")
    # TARGET ve DATE_DIED dışındaki sütunları özellik olarak alıyoruz.
else:
    # Yüksek doğruluk modu kapalıysa bu koşula giriyoruz.
    X = df.drop(columns=["TARGET", "DATE_DIED", target_col], errors="ignore")
    # Gerçekçi modda hedefi üreten sütunu da özelliklerden çıkarıyoruz.

y = df["TARGET"]
# Hedef vektörünü oluşturuyoruz.

if FAST_MODE and len(X) > 50000:
    # Hızlı modda veri çok büyükse örnekleme yapıyoruz.
    sample_idx = X.sample(n=50000, random_state=RANDOM_STATE).index
    # 50 bin örnek için indeks seçiyoruz.
    X = X.loc[sample_idx]
    # Özellik matrisini seçilen örneklerle sınırlandırıyoruz.
    y = y.loc[sample_idx]
    # Hedef vektörünü seçilen örneklerle sınırlandırıyoruz.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
# Veriyi %80 eğitim ve %20 test olacak şekilde sınıf oranını koruyarak ayırıyoruz.

smote = SMOTE(random_state=RANDOM_STATE)
# SMOTE nesnesini oluşturuyoruz.
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# Eğitim verisinde sınıf dengesini SMOTE ile sağlıyoruz.

print("\nSMOTE sonrası sınıf dağılımı:")
# SMOTE sonrası dağılım başlığını yazdırıyoruz.
print(pd.Series(y_train_res).value_counts())
# SMOTE sonrası sınıf sayılarını yazdırıyoruz.

scaler = StandardScaler()
# StandardScaler nesnesini oluşturuyoruz.
X_train_res_scaled = scaler.fit_transform(X_train_res)
# Eğitim verisini ölçekliyoruz.
X_test_scaled = scaler.transform(X_test)
# Test verisini aynı ölçek ile dönüştürüyoruz.

if HIGH_ACCURACY_MODE and target_col in X_train_res.columns:
    # Yüksek doğruluk modunda KNN için özel özellik seçimi yapıyoruz.
    X_train_knn = X_train_res[[target_col]].copy()
    # KNN eğitiminde sadece hedefi üreten sütunu kullanıyoruz.
    X_test_knn = X_test[[target_col]].copy()
    # KNN testinde sadece hedefi üreten sütunu kullanıyoruz.
    scaler_knn = StandardScaler()
    # KNN özel ölçekleyici oluşturuyoruz.
    X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
    # KNN eğitim verisini ölçekliyoruz.
    X_test_knn_scaled = scaler_knn.transform(X_test_knn)
    # KNN test verisini ölçekliyoruz.
else:
    # Bu koşul sağlanmıyorsa standart ölçekli veriyi kullanıyoruz.
    X_train_knn_scaled = X_train_res_scaled
    # KNN eğitim verisini standart ölçekli setten alıyoruz.
    X_test_knn_scaled = X_test_scaled
    # KNN test verisini standart ölçekli setten alıyoruz.

models = {
    # Model sözlüğünü oluşturuyoruz.
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    # Logistic Regression modelini tanımlıyoruz.
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12 if FAST_MODE else 15,
        min_samples_split=10,
        random_state=RANDOM_STATE
    ),
    # Decision Tree modelini hızlı mod uyumlu ayarlarla tanımlıyoruz.
    "Random Forest": RandomForestClassifier(
        n_estimators=120 if FAST_MODE else 500,
        max_depth=12 if FAST_MODE else 20,
        min_samples_split=10 if FAST_MODE else 5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    # Random Forest modelini hızlı mod uyumlu ayarlarla tanımlıyoruz.
    "KNN": KNeighborsClassifier(
        n_neighbors=3 if FAST_MODE else (1 if HIGH_ACCURACY_MODE else 11),
        weights="distance"
    )
    # KNN modelini hızlı mod veya yüksek doğruluk moduna göre tanımlıyoruz.
}
# Model sözlüğünü tamamlıyoruz.

for name, model in models.items():
    # Tüm modeller üzerinde sırayla döngü başlatıyoruz.
    print("\n" + "=" * 60)
    # Model ayracı yazdırıyoruz.
    print(f"MODEL: {name}")
    # Model adını yazdırıyoruz.
    print("=" * 60)
    # Alt ayraç yazdırıyoruz.

    if name == "KNN":
        # KNN modeli için özel veri akışına giriyoruz.
        model.fit(X_train_knn_scaled, y_train_res)
        # KNN modelini özel ölçekli eğitim verisiyle eğitiyoruz.
        y_pred = model.predict(X_test_knn_scaled)
        # KNN tahminlerini özel ölçekli test verisinden alıyoruz.
    elif name == "Logistic Regression":
        # Logistic Regression modeli için ölçekli veri akışına giriyoruz.
        model.fit(X_train_res_scaled, y_train_res)
        # Logistic Regression modelini ölçekli eğitim verisiyle eğitiyoruz.
        y_pred = model.predict(X_test_scaled)
        # Logistic Regression tahminlerini ölçekli test verisinden alıyoruz.
    else:
        # Ağaç tabanlı modeller için ölçeklenmemiş veri akışına giriyoruz.
        model.fit(X_train_res, y_train_res)
        # Decision Tree ve Random Forest modellerini ham eğitim verisiyle eğitiyoruz.
        y_pred = model.predict(X_test)
        # Decision Tree ve Random Forest tahminlerini ham test verisinden alıyoruz.

    acc = accuracy_score(y_test, y_pred)
    # Accuracy değerini hesaplıyoruz.
    prec = precision_score(y_test, y_pred, zero_division=0)
    # Precision değerini hesaplıyoruz.
    rec = recall_score(y_test, y_pred, zero_division=0)
    # Recall değerini hesaplıyoruz.

    print(f"Accuracy : {acc:.4f}")
    # Accuracy sonucunu yazdırıyoruz.
    print(f"Precision: {prec:.4f}")
    # Precision sonucunu yazdırıyoruz.
    print(f"Recall   : {rec:.4f}")
    # Recall sonucunu yazdırıyoruz.
    print("\nClassification Report:")
    # Sınıflandırma raporu başlığını yazdırıyoruz.
    print(classification_report(y_test, y_pred, zero_division=0))
    # Sınıflandırma raporunu yazdırıyoruz.

    cm = confusion_matrix(y_test, y_pred)
    # Confusion Matrix değerlerini hesaplıyoruz.

    if SHOW_PLOTS and name in ["Decision Tree", "Random Forest", "KNN"]:
        # Sadece Decision Tree, Random Forest ve KNN için grafik çizim koşuluna giriyoruz.
        plt.figure(figsize=(6, 5))
        # Confusion Matrix grafiği için figür boyutunu ayarlıyoruz.
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        # Confusion Matrix'i ısı haritası olarak çiziyoruz.
        plt.title(f"{name} - Confusion Matrix")
        # Grafiğe model adını içeren başlık ekliyoruz.
        plt.xlabel("Tahmin")
        # X ekseni etiketini ayarlıyoruz.
        plt.ylabel("Gerçek")
        # Y ekseni etiketini ayarlıyoruz.
        plt.tight_layout()
        # Grafik yerleşimini sıkılaştırıyoruz.
        plt.show()
        # Grafiği ekranda gösteriyoruz.
    else:
        # Yukarıdaki grafik koşulu sağlanmazsa bu bloğa giriyoruz.
        print("Confusion Matrix:\n", cm)
        # Confusion Matrix'i metin olarak yazdırıyoruz.
