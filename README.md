#  **ML Covid-19 Hasta Tahmini**
---

## **1. Proje Açıklaması**
Bu proje, kaggle dan indirdiğimiz veri setini analiz ederek **COVID-19 risk durumunu(pozitif/negatif)** yüksek doğrulukla tahmin
eden bir makine öğrenmesi çalışmasıdır.

---
##  **2. Veri Seti Tanıtımı**
Projede kullanılan veri seti, Meksika Hükümeti tarafından paylaşılan anonimleştirilmiş gerçek hasta verilerini içerir.
* **Veri Seti Kaynağı:** [Kaggle - COVID-19 Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)
* **İçerik:** Cinsiyet, yaş, kronik hastalıklar (diyabet, astım, obezite vb.), hastaneye yatış durumu gibi **20'den fazla klinik özellik**
*  bulunmaktadır.
---
##  **3. Veri Ön İşleme Adımları**
Verinin modele hazır hale getirilmesi için uygulanan teknikler:
* **Sütun Temizliği:** Tüm sütun isimleri standart hale getirilmiş ve büyük harfe çevrilmiştir.
* **Hedef Değişken (Target):** `CLASIFFICATION_FINAL` sütunu baz alınarak; **1, 2 ve 3** değerleri **"Pozitif (1)"**, diğer değerler
*  **"Negatif (0)"** olarak gruplanmıştır.
* **Eksik Veri Yönetimi:** Bilinmeyen değerler (`97, 98, 99`) `NaN` yapılmış ve **Medyan** yöntemiyle doldurulmuştur.
* **Aykırı Değer (Outlier):** Yaş (`AGE`) sütununda uç değerler **IQR yöntemi** ile temizlenmiştir.
* **Sınıf Dengelenmesi (SMOTE):** Eğitim verisindeki sınıflar arası sayısal fark, **SMOTE** algoritması ile eşitlenmiştir.
* **Ölçeklendirme:** Veriler **StandardScaler** ile standartlaştırılmıştır.
---
##  **4. Kullanılan Algoritmaların Mantığı**
Projede dört farklı temel sınıflandırma algoritması test edilmiştir:
---
1.  **Logistic Regression:** Değişkenler arasındaki ilişkiyi olasılıksal bir fonksiyonla modeller.
2.  **Decision Tree:** Veriyi belirli kriterlere göre dallara ayırarak karar kuralları oluşturur.
3.  **Random Forest:** Birden fazla karar ağacını aynı anda eğiterek en iyi sonucu seçen bir **Topluluk (Ensemble)** yöntemidir.
4.  **K-Nearest Neighbors (KNN):** Veri noktasını, en yakınındaki komşularının çoğunluk sınıfına göre atayan bir modeldir.
---
##  **5. Model Performans Karşılaştırması**
Aşağıdaki tablo, modellerin test verisi üzerindeki başarı metriklerini göstermektedir:
| **Algoritma** | **Doğruluk (Accuracy)** | **Keskinlik (Precision)** | **Duyarlılık (Recall)** |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | %65 - %75 | Orta | Orta |
| **Decision Tree** | %85 - %90 | **Yüksek** | **Yüksek** |
| **Random Forest** | **%90+** | **Çok Yüksek** | **Çok Yüksek** |
| **KNN (Özel Mod)** | **%95+** | **Çok Yüksek** | **Çok Yüksek** |
---
##  **6. Sonuç ve Yorumlar**
* **En Başarılı Model:** Projede en yüksek başarımı **Random Forest** ve **KNN** göstermiştir.
* **Duyarlılık (Recall):** Sağlık verilerinde "hasta" olanı tespit etmek kritik olduğundan, **SMOTE** ile duyarlılık oranı maksimize
*  edilmiştir.
* **Veri Sızıntısı:** Ölüm tarihi gibi sonucu doğrudan etkileyen veriler temizlenerek modelin gerçekçi tahminde bulunması sağlanmıştır.
<img width="596" height="496" alt="korona1" src="https://github.com/user-attachments/assets/41cf3a30-5e55-441a-a155-b3f29627e062" />

**Şekil 1:** Veri setindeki sınıf dağılımını göstermektedir. Pozitif ve negatif vakaların oranı gözlemlenerek veri dengesizliği (class imbalance) problemi analiz edilmiştir.

---

<img width="579" height="493" alt="korona2" src="https://github.com/user-attachments/assets/0cfedd49-456d-4557-b9c2-178c0ef78910" />

**Şekil 2:** Modellerin doğruluk (Accuracy) karşılaştırmasını göstermektedir. Random Forest ve KNN modellerinin diğer algoritmalara göre daha yüksek performans sergilediği görülmektedir.

---

<img width="595" height="500" alt="korona3" src="https://github.com/user-attachments/assets/d71f5fbd-a469-4e1e-906b-6b91a26ab611" />

**Şekil 3:** Confusion Matrix (Karışıklık Matrisi) çıktısını göstermektedir. Modelin doğru ve yanlış sınıflandırma sayıları analiz edilerek özellikle Recall değerinin sağlık verileri açısından önemi değerlendirilmiştir.

---

##  **7. Kodların Nasıl Çalıştırılacağı**
1.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
    ```
2.  **Veri Setini Hazırlayın:** Kaggle'dan indirilen `Covid Data.csv` dosyasını proje klasörüne ekleyin.
3.  **Çalıştırın:**
    ```bash
    python main.py
    ```

---

##  **Hazırlayan**

**İbrahim Türköz**

**25019921056**

**Makine Öğrenmesi Dersi**

**Covid-19 Hasta Tahmini Ödevi**
