#  Chess Match Outcome Prediction — SVM

Bu proje, satranç maçlarının meta verilerini kullanarak beyaz mı, siyah mı yoksa oyun berabere mi biter tahmini yapan bir makine öğrenimi çalışmasıdır. Projede veri ön işleme, özellik mühendisliği, görselleştirme, çoklu model denemeleri, model karşılaştırması ve en iyi model ile tahmin yapma adımları uygulanmıştır.

##  Proje Amacı

Bu çalışmanın asıl amacı, satranç oyunlarında **kazanan tarafı tahmin etmek** için denetimli öğrenme modellerini kullanarak hepsinin sonuç analizlerinin karşılaştırılması neticesinde en iyi accuracy değerini veren modeli bulmak olacaktır.
Model;

* **white_rating**
* **black_rating**
* **rating_diff**
* **time_control**
* **opening (ilk hamle)**
  gibi özelliklerden yola çıkarak “winner” değerini tahmin eder.

## Kullanılan Kütüphaneler

Projede kullanılan temel Python kütüphaneleri:

```python
pandas, numpy, matplotlib, seaborn
scikit-learn (LabelEncoder, LogisticRegression,RandomForestClassifier,GaussianNB, KNeighborsClassifier,StandardScaler,SVC, DecisionTreeClassifier, classification_report, accuracy_score,train_test_split,confusion_matrix,roc_curve, roc_auc_score)
```

## Veri Seti Açıklaması

Veri Kaggle üzerinde yayınlanan bir satranç oyunları veri setidir.

Projede kullanılan önemli sütunlar:

* **white_rating**: Beyaz oyuncunun ratingi
* **black_rating**: Siyah oyuncunun ratingi
* **winner**: Kazanan taraf (“white”, “black”, “draw”)
* **pgn**: Oyunun hamlelerini içeren PGN formatı
* **time_control**: Oyun süresi
Bu projede pgn içinden sadece ilk hamle (opening) çıkarılmıştır.
##  Veri Ön İşleme

1. **Başlamamış oyunların kaldırılması**
   <br>
   <br>
   status sütunu içerisindeki noStart ile eşleşen satırlar veriden kaldırılır.

```python
df = df[df['status'] != 'noStart']
```

2. **Eksik değerlerin silinmesi**
   <br>
   <br>
   Kullanacağımız sütunlarda NaN olan değerler var ise bunlar temizlenir. 

```python
df = df.dropna(subset=['white_rating', 'black_rating', 'winner', 'pgn', 'time_control'])
```

3. **25000 satırlık örnek seçimi**
   <br>
   <br>
   Büyük veri modelleri gereksiz yavaşlatmaması için dataset rastgele 25.000 satıra indirildi.
```python
df = df.sample(n=25000, random_state=42)
```

## Özellik Mühendisliği
1. **İlk hamle (Opening) çıkarımı**
<br>

pgn formatı komple hamle metni içerir.

Ancak tüm hamleleri kullanmak model için gereksiz karmaşıktır. Bu yüzden:
```python
def simplify_opening(pgn):
    moves = pgn.strip().split()
    return moves[0] if len(moves) > 0 else "Unknown"

df["Opening"] = df["pgn"].apply(simplify_opening)
```
Böylece sadece başlangıç hamlesi alındı.
Bu, açılışın oyun sonucunu etkileyebileceği hipotezine dayanır.
Bu işlem daha sonra Label Encoding ile sayısal hale getirilir.
<br>
<br>
2. **Rating Farkı (rating_diff) – En güçlü feature**
<br>
<br>
Bu özellik beyazın siyaha göre ne kadar güçlü olduğunu gösterir:

rating_diff = white_rating - black_rating

Bu değişken sonuç üzerinde çok kritiktir:

Pozitif → Beyaz güçlü

Negatif → Siyah güçlü

0 → Oyuncular eşit güçte → Berabere ihtimali artar

## Veri Görselleştirme

Projede yapılan başlıca grafikler:

### Kazanan Dağılımı

* Hangi tarafın daha çok kazandığını gösterir. Yani White, black ve draw sayılarını gösteren bar plot diyebiliriz.

### Rating Dağılımı

* Beyaz ve siyah oyuncu rating histogramları.

### Rating farkı – kazanan ilişkisi

Her sınıf için ayrı histogram çizildi.
Gözlem:

rating_diff pozitifse beyaz kazanma olasılığı yükseliyor

Negatifse siyah avantajlı

0 civarı → draw artıyor

### En Popüler 10 Açılış

* İlk hamleye göre sıralanmış açılış popülerlik grafiği.
* e4, d4, Nf3, c4 öne çıkıyor.

## Veri Kodlama ve Ölçekleme
* **Label Encoding**

    String değişkenler sayısallaştırıldı:

    -> time_control

    -> Opening

    -> winner (target)

```python
le_opening.fit_transform(...)
le_time.fit_transform(...)
```
* **StandardScaler**

    Özellikle Logistic Regression, SVM ve KNN gibi modeller için ölçekleme çok önemlidir.

    O nedenle:

    Eğitim seti ile fit()

    Test ve gerçek tahminde transform()

    yapılmıştır.

## Eğitilen Modeller
Model için kullanılan özellikler:

```python
features = ["white_rating", "black_rating", "time_control", "Opening"]
target = "winner"
```
Model eğitimi:
<br>
Modeli eğitmek için veri setimizi %80 eğitim ve %20 test için olmak üzere parçalıyoruz.Her model aynı train-test split üzerinde denendi.


```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True) # ROC Curve için probability=True lazım
}
```

Aşağıdaki 6 model denenmiştir:

Model	Açıklama
Logistic Regression	Lineer sınıflandırıcı, baseline
Random Forest	Ensemble, çok güçlü model
Decision Tree	Basit ama genelde aşırı öğrenir
KNN	Komşuluk tabanlı
Naive Bayes	Basit, hızlı
SVM	Güçlü, özellikle scaled veride iyi
## Modellerin Başarı Sonuçları

Logistic Regression Başarısı: %74.22
Random Forest Başarısı: %70.94
Decision Tree Başarısı: %65.88
KNN Başarısı: %71.32
Naive Bayes Başarısı: %74.34
SVM Başarısı: %74.42

En yüksek doğruluk SVM (Support Vector Machine) ile elde edilmiştir:
EN İYİ MODEL: SVM (%74.42)

## Sonuçların Analizi ve SVM’nin Üstünlüğü

* **Veri yapısı ve lineer ayrılabilirlik**

    Satranç maç sonuçları rating_diff ve white_rating/black_rating gibi lineer etkili değişkenlerle kısmen ayrılabilir.

    Logistic Regression de iyi bir sonuç verdi (%74.22) fakat SVM, margin (marjin) kullanarak sınıf sınırını optimize etti.

* **Scaling ve kernel avantajı**
    Veriler StandardScaler ile ölçeklendi. SVM, özellikle scaled verilerde çok daha stabil ve yüksek doğruluk sağlar.

    KNN ve Decision Tree, scaling veya noisy verilerden etkilenir; bu yüzden biraz daha düşük performans sergiledi.

* **Overfitting riskleri**

    Random Forest ve Decision Tree, küçük veri setlerinde overfit olma eğilimindedir (%65–70 civarı).

    SVM, margin maximization ile aşırı öğrenmeyi önler ve genelleme kapasitesi yüksektir.

* **Naive Bayes performansı**

    NB varsayımsal olarak feature’ların birbirinden bağımsız olduğunu varsayar.

    Oysa rating_diff ve white_rating gibi özellikler bağımlı olduğundan performans çok düşük kalmaz ama SVM yine biraz daha iyi çıkar.

Özet:

En yüksek doğruluk SVM ile elde edilmiştir: %74.42

SVM tercih sebebi:

  * Verinin lineer olmayan sınırlarını iyi ayırabilmesi

  * Scaling sonrası daha kararlı ve yüksek performans

  * Overfitting’e karşı dayanıklı yapısı

Bu nedenle projede tahmin ve ileri analizlerde SVM modeli kullanılmalıdır.
## Model Değerlendirme

Modelin başarı ölçütleri:

* **Accuracy Score**
* **Classification Report**
* **Confusion Matrix**

## ROC ve Çok Sınıflı F1 Analizi

Veri seti **3 sınıf** içerdiğinden klasik ROC eğrisi yalnızca binary veri için çalışır.

Kod:

```python
if len(le_y.classes_) == 2:
    ...
else:
    çok sınıflı F1-score karşılaştırma grağiği
```

Bu nedenle projede 3 sınıf için:

  -> F1-score
  
  -> Sınıf bazlı başarı dağılımı

gösterilmiştir.

## Gerçek Senaryo Testi
Modelin başarısını kanıtlamak için eğitim setinde olmayan, tamamen yapay bir maç senaryosu oluşturulmuş ve modele sorulmuştur.

Senaryo:

Beyaz Oyuncu: 2080 Elo (Ortalama oyuncu)

Siyah Oyuncu: 2928 Elo (Grandmaster seviyesi - Çok güçlü)

Açılış: e4 (King's Pawn)

Beklenti: Siyah oyuncunun çok yüksek puan farkı nedeniyle maçı kazanması.
 ```python
#=== EN İYİ MODEL TAHMİNİ ===
En iyi model: SVM
Model Tahmini: Black
```

## Sonuç

Bu proje, satranç oyuncularının ratingleri ve oyun açılış hareketi gibi bilgiler kullanılarak maç sonucunu tahmin eden geniş kapsamlı uygulamadır. Veriyi her bir model için eğitmiş ve bunların sonuçlarını analiz ederek en iyi modeli seçmiştir. Veri temizleme, etiketleme, model eğitimi ve performans görselleştirmelerinin tamamı ayrıntılı bir şekilde yapılmıştır. 



