# Drone Detection with CNN and ResNet18

Bu proje, farklı boyut, şekil ve renklere sahip drone'ların tespit edilmesi amacıyla yaklaşık **18.000 görüntüden oluşan Drone Detection Dataset** üzerinde iki farklı derin öğrenme yaklaşımının geliştirilmesini ve karşılaştırılmasını kapsamaktadır.

Projede iki farklı nesne tespit yöntemi uygulanmıştır:

1. **Sıfırdan geliştirilen CNN + Sliding Window + Image Pyramid**
2. **ResNet18 + Fully Convolutional Network (FCN) + Grid-based Detection**

Her iki yöntemde de model mimarisi, veri ön işleme, hiperparametre optimizasyonu ve değerlendirme süreçleri ayrı olarak ele alınmıştır.

> Projeye dair daha ayrıntılı bilgi edinmek için **`Rapor.pdf`** dosyasında bulunan proje raporuna bakabilirsiniz. 


---

## Veri Seti

Kullanılan veri setinde:

* Drone içermeyen görüntüler
* Tek drone içeren görüntüler
* Birden fazla drone içeren görüntüler
* Büyük ve küçük drone örnekleri
* Drone'a benzeyebilecek uçak, kuş ve balon gibi nesneler

bulunmaktadır.

Veri seti kaynak tarafından üç parçaya ayrılmıştır ve projede aynı ayrım korunmuştur:

| Veri Seti  | Görüntü Sayısı |
| ---------- | -------------: |
| Eğitim     |         11.577 |
| Geçerleme  |          3.952 |
| Test       |          2.597 |
| **Toplam** |     **18.126** |

### Veri Seti Atıfı

**Drone Detection Dataset** — Aatish Kumar Sahu. *Roboflow Universe*, 2025.

[Drone Detection Dataset – Roboflow Universe](https://universe.roboflow.com/aatish-kumar-sahu-57emd/drone-detection-1ghph)

---

# Yöntem 1 — Custom CNN + Sliding Window

İlk yöntemde herhangi bir önceden eğitilmiş model kullanılmamış, veri seti üzerinde sıfırdan bir **CNN + Dense Head** mimarisi geliştirilmiştir.

Bu yöntemin temel amacı, özellikle farklı boyutlarda ve görüntü içerisinde farklı alan kaplayan drone'ları tespit edebilen merkezi bir drone tanıma modeli oluşturmaktır.

## Ön İşleme

Training aşamasında drone bounding box'ları kullanılarak pozitif ve negatif örnekler oluşturulmuştur.

### Pozitif Örnekler

Her drone bounding box'ı:

* Dış tarafına rastgele padding eklenerek,
* Rastgele x/y jitter uygulanarak,
* Drone'u içeren bölge crop edilerek

224×224 boyutuna yeniden ölçeklendirilmiştir.

Bu işlem sayesinde modelin farklı konum ve boyutlardaki drone'ları öğrenmesi amaçlanmıştır.

### Negatif Örnekler

Pozitif örneklerle aynı miktarda rastgele görüntü bölgeleri seçilmiştir.

Seçilen bölgelerin drone bounding box'larıyla IoU değerleri kontrol edilmiştir. Belirlenen `NEG_IOU_THRESHOLD` değerinden daha fazla drone içeren bölgeler negatif örneklerden çıkarılmıştır.

Geriye kalan görüntü bölgeleri:

```text
label = 0
confidence = 0
```

şeklinde negatif örnek olarak kullanılmıştır.

---

## CNN Mimarisi

En başarılı modelde 5 convolution block kullanılmıştır.

```text
Input: 224 × 224

Conv Block 1
32 filters
3 × 3 Conv
BatchNorm
Leaky ReLU
2 × 2 MaxPool

        ↓

Conv Block 2
64 filters
3 × 3 Conv
BatchNorm
Leaky ReLU
2 × 2 MaxPool

        ↓

Conv Block 3
128 filters
3 × 3 Conv
BatchNorm
Leaky ReLU
2 × 2 MaxPool

        ↓

Conv Block 4
256 filters
3 × 3 Conv
BatchNorm
Leaky ReLU
2 × 2 MaxPool

        ↓

Conv Block 5
512 filters
3 × 3 Conv
BatchNorm
Leaky ReLU
2 × 2 MaxPool

        ↓

7 × 7 × 512 Feature Map

        ↓

Flatten
25,088 features

        ↓

Dense 128
Leaky ReLU
Dropout 0.3

        ↓

Dense 128
Leaky ReLU
Dropout 0.3

        ↓

Output: 5
```

Çıkış katmanındaki 5 değer:

```text
[confidence, x, y, w, h]
```

şeklindedir.

* `confidence`: Patch içerisinde drone bulunma olasılığı
* `x, y`: Drone bounding box merkez koordinatları
* `w, h`: Bounding box genişliği ve yüksekliği

---

## Multi-Task Loss

Model aynı anda hem drone sınıflandırması hem de bounding box regresyonu gerçekleştirdiğinden iki farklı kayıp fonksiyonu kullanılmıştır.

### Confidence Loss

Standart BCE yerine **Focal Loss tabanlı BCE** kullanılmıştır.

Bunun temel nedeni sliding-window yaklaşımında çok fazla background patch oluşmasıdır.

Modelin tüm patch'leri background olarak tahmin ederek yüksek accuracy elde etmesini engellemek amacıyla:

* Focal Loss
* `pos_weight`

kullanılmıştır.

### Regression Loss

Bounding box koordinatları için:

```text
MSE Loss
```

kullanılmıştır.

Toplam kayıp:

```text
Total Loss = Confidence Loss + Regression Loss
```

şeklinde oluşturulmuştur.

---

# Inference — Image Pyramid + Sliding Window

Training sırasında model 224×224 drone patch'leri üzerinde eğitildiği için inference aşamasında farklı boyutlardaki drone'ları tespit edebilmek amacıyla **image pyramid** kullanılmıştır.

Kullanılan ölçekler:

```text
0.5
1.0
1.5
```

Her ölçeklendirilmiş görüntü üzerinde:

```text
224 × 224 Sliding Window
```

gezdirilerek tahminler alınmıştır.

Sliding window sonucunda aynı drone için birden fazla bounding box üretilebildiğinden sonuçlara **Non-Maximum Suppression (NMS)** uygulanmıştır.

---

# Yöntem 1 — Hiperparametre Optimizasyonu

Hiperparametre seçiminde temel olarak validation F1 ve mAP50 değerleri karşılaştırılmıştır.

Deneylerde özellikle aşağıdaki parametreler incelenmiştir:

* Learning rate
* Convolution block sayısı
* Başlangıç filtre sayısı
* Dense head mimarisi
* Dropout
* Optimizer
* Fine-tuning stratejisi

### Elde Edilen Bulgular

**Learning Rate**

`1e-4` learning rate değerinin `1e-5` değerine göre daha başarılı olduğu gözlemlenmiştir.

**Convolution Block**

5 convolution block kullanılması, 4 convolution block kullanımına göre daha başarılı sonuç vermiştir.

5 block sonrasında feature map boyutu 7×7 seviyesine düştüğü için daha fazla pooling uygulanmamıştır.

**Dense Head**

128 → 128 yapısındaki dense head başarılı sonuç vermiştir.

**Dropout**

Deneylerde dropout kullanımının performansı artırmadığı gözlemlenmiş ve optimal yapı için dropout kullanılmaması tercih edilmiştir.

---

# Yöntem 2 — ResNet18 + Fully Convolutional Detector

İkinci yöntemde önceden eğitilmiş **ResNet18** modeli backbone olarak kullanılmıştır.

Bu yöntemde temel fark, sliding window kullanılmamasıdır.

Model bütün görüntüyü tek seferde işleyerek görüntüyü **32×32 grid hücrelerine** ayırmakta ve her grid için drone bulunma olasılığı ile bounding box tahmini gerçekleştirmektedir.

Bu yaklaşım sayesinde inference aşamasında binlerce ayrı patch'in modele verilmesi gerekmemektedir.

---

## Veri Ön İşleme

Büyük görüntülerin GPU belleğini aşırı kullanmasını önlemek amacıyla görüntüler aspect ratio korunarak yeniden boyutlandırılmıştır.

Maksimum boyut:

```text
256 × 256
```

olarak belirlenmiştir.

Ardından görüntüler:

1. `[0, 1]` aralığına normalize edilmiş,
2. ImageNet mean/std değerleri ile normalize edilmiştir.

Bu işlem ResNet18'in önceden eğitildiği veri dağılımıyla uyum sağlamak amacıyla uygulanmıştır.

---

# Fully Convolutional Detection Head

ResNet18'in CNN katmanları backbone olarak kullanılmıştır.

Üzerine aşağıdaki detection head eklenmiştir:

```text
ResNet18 Backbone
       ↓
CNN 256
3 × 3
ReLU
       ↓
CNN 128
3 × 3
ReLU
       ↓
CNN 64
3 × 3
ReLU
       ↓
Output CNN
5 channels
1 × 1
```

Çıkıştaki 5 değer:

```text
[confidence, x, y, w, h]
```

şeklindedir.

### Aktivasyonlar

`confidence`, `x` ve `y`:

```text
Sigmoid
```

ile 0–1 aralığına getirilmiştir.

`w` ve `h`:

```text
Softplus
```

ile pozitif değerler olacak şekilde sınırlandırılmıştır.

---

# Grid-Based Detection

Model görüntüyü 32×32 grid hücrelerine ayırmaktadır.

Bir drone'un merkezi hangi grid içerisinde bulunuyorsa yalnızca o grid pozitif olarak etiketlenmektedir.

Bu yaklaşım sayesinde aynı obje için birden fazla grid'in pozitif tahmin üretmesinin önüne geçilmesi amaçlanmıştır.

Bounding box koordinatları grid'e göre normalize edilmiştir:

```text
x → grid width'e göre normalize
y → grid height'e göre normalize
w → grid width'e göre normalize
h → grid height'e göre normalize
```

Bu sayede farklı boyutlardaki bounding box'ların loss üzerindeki etkisinin dengelenmesi hedeflenmiştir.

---

# Loss Function

Toplam loss iki ana bileşenden oluşmaktadır:

```text
Total Loss = Classification Loss + Localization Loss
```

### Classification Loss

Binary classification problemi olduğu için:

```text
Binary Cross Entropy
```

kullanılmıştır.

### Localization Loss

Aşağıdaki loss fonksiyonları deneylerde karşılaştırılmıştır:

* MSE
* Smooth L1
* L1

En başarılı sonuç:

```text
L1 Loss
```

ile elde edilmiştir.

Ayrıca localization loss toplam loss'a dahil edilmeden önce 5 katsayısı ile çarpılmıştır:

```text
Total Loss = BCE Loss + 5 × L1 Loss
```

---

# Transfer Learning ve Fine-Tuning

ResNet18 backbone için farklı eğitim stratejileri denenmiştir.

### Aşama 1

ResNet18 tamamen dondurulmuş ve yalnızca detection head eğitilmiştir.

```text
ResNet18 → Frozen
Detection Head → Trainable
```

### Aşama 2

Daha başarılı sonuçlar elde etmek amacıyla ResNet18'in son iki katmanı açılmış ve detection head ile birlikte fine-tune edilmiştir.

```text
ResNet18
└── Last 2 Layers → Trainable

Detection Head → Trainable
```

İki aşamalı eğitimde:

* İlk aşama: 60 epoch
* İkinci aşama: maksimum 60 epoch
* Early stopping: ikinci aşamada aktif

olarak uygulanmıştır.

---

# Hiperparametre Optimizasyonu

Çeşitli deneylerde aşağıdaki parametreler karşılaştırılmıştır:

* Detection head derinliği
* Dropout
* Learning rate
* Optimizer
* Fine-tuning stratejisi
* Localization loss
* Activation function

### Optimal Konfigürasyon

| Parametre            | Optimal Değer         |
| -------------------- | --------------------- |
| Backbone             | ResNet18              |
| Optimizer            | AdamW                 |
| Learning Rate        | 0.0001                |
| Detection Head       | 256 → 128 → 64        |
| Activation           | ReLU                  |
| Dropout              | Yok                   |
| Localization Loss    | L1                    |
| Localization Weight  | 5                     |
| Fine-Tuning          | ResNet18 son 2 katman |
| Objectness Threshold | 0.5                   |
| Training Stage 1     | 60 epoch              |
| Training Stage 2     | Max. 60 epoch         |

---

# Değerlendirme Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilmiştir:

* Precision
* Recall
* F1 Score
* mAP@50

Bounding box eşleşmesinde:

```text
IoU ≥ 0.50
```

eşik değeri kullanılmıştır.

Tahminler objectness/confidence skoruna göre sıralanmıştır. Daha yüksek confidence değerine sahip tahminler öncelikli olarak değerlendirilmiş ve uygun IoU değerine sahip ground-truth bounding box'larla eşleştirilmiştir.

Bu problemde **True Negative (TN)** değerinin anlamlı ve ölçülebilir olmaması nedeniyle accuracy metriği kullanılmamıştır.

Bu nedenle temel değerlendirme metrikleri:

```text
mAP@50
F1
Precision
Recall
```

olarak belirlenmiştir.

---

# Yöntemlerin Karşılaştırılması

| Özellik             | Yöntem 1       | Yöntem 2      |
| ------------------- | -------------- | ------------- |
| Backbone            | Custom CNN     | ResNet18      |
| Pretrained Model    | Hayır          | Evet          |
| Detection           | Sliding Window | Grid-based    |
| Image Pyramid       | Var            | Yok           |
| Fully Convolutional | Hayır          | Evet          |
| Dense Layer         | Var            | Yok           |
| Bounding Box        | Var            | Var           |
| NMS                 | Var            | Gerekli değil |
| Inference           | Daha yavaş     | Daha hızlı    |
| Transfer Learning   | Yok            | Var           |

---

# Yöntem 1 — Sonuçlar ve Problemler

Custom CNN + Sliding Window yaklaşımı temel olarak çalışmasına rağmen inference aşamasında çok sayıda patch işlenmesi gerektiği için önemli problemler ortaya çıkmıştır.

En önemli problemlerden biri yüksek **False Positive** oranıdır.

Tek bir görüntü üzerinde binlerce window değerlendirilmesi sonucunda modelin patch başına başarımı iyi olsa dahi birkaç hatalı tahmin toplam görüntü performansını ciddi şekilde düşürmektedir.

Bu durum özellikle:

```text
mAP@50
F1
```

metriklerinde belirgin bir düşüşe neden olmuştur.

Yöntemin mAP@50 performansı yaklaşık **%17 seviyesinde** kalmıştır.

### Diğer problemler

* Drone sürülerinde NMS'in agresif çalışması
* Birbirine yakın drone'ların bastırılması
* Farklı renk ve dokulara sahip drone'larda confidence düşmesi
* Tek tip anchor yaklaşımının sınırlamaları
* Sliding window nedeniyle yüksek inference maliyeti
* Çok sayıda background window nedeniyle False Positive oluşması

---

# Yöntem 1 — Potansiyel Geliştirmeler

### Anchor Box Kullanımı

Farklı boyut ve aspect ratio'lara sahip anchor box'lar kullanılarak özellikle drone sürülerindeki detection problemlerinin azaltılması hedeflenebilir.

### Color Augmentation

Drone renk ve doku çeşitliliğini artırmak için daha kapsamlı color augmentation uygulanabilir.

Alternatif olarak grayscale fine-tuning gerçekleştirilebilir.

### Dynamic Tiling

Sabit image pyramid ve sliding window yerine görüntü içerisindeki nesne yoğunluğu veya çekim yüksekliği gibi faktörlere göre pencere boyutunu dinamik olarak belirleyen yöntemler kullanılabilir.

Bu amaçla **SAHI benzeri slicing/tiling yaklaşımları** değerlendirilebilir.

---

# Yöntem 2 — Sonuçlar ve Problemler

ResNet18 tabanlı Fully Convolutional yaklaşım, sliding window kullanmaması nedeniyle inference hızı açısından Yöntem 1'e göre avantaj sağlamaktadır.

Model bütün görüntüyü tek seferde işleyerek grid tabanlı tahmin gerçekleştirmektedir.

Bununla birlikte yöntemin bazı sınırlamaları bulunmaktadır:

* Donanım kısıtları nedeniyle görüntüler 256×256 boyutunu aşmayacak şekilde resize edilmiştir.
* Bazı küçük drone örneklerinde tespit başarımı düşmektedir.
* Drone'ların renk ve şekil çeşitliliği bazı örneklerde detection problemlerine neden olmaktadır.
* Daha büyük görüntülerle çalışmak mevcut donanımda bellek problemi oluşturabilmektedir.

Model kuş ve balon gibi drone dışı uçan nesneleri ayırt edebilmesine rağmen bazı drone örneklerini tespit edememiştir.

---

# Yöntem 2 — Potansiyel Geliştirmeler

### Daha Büyük Görüntüler

Daha güçlü donanım kullanılarak daha yüksek çözünürlüklü görüntülerle eğitim gerçekleştirilebilir.

### Quantization

Model quantization uygulanarak bellek kullanımı azaltılabilir ve daha büyük görüntülerle çalışma imkanı sağlanabilir.

### Pipeline Parallelism

Dağıtık model eğitimi ve pipeline parallelism kullanılarak yüksek çözünürlüklü görüntülerle eğitim gerçekleştirilebilir.

### Daha Büyük Veri Seti

Drone çeşitliliğini artırmak amacıyla daha geniş ve çeşitli bir veri seti kullanılabilir.

---

# Genel Değerlendirme

Bu projede drone detection problemi iki farklı derin öğrenme yaklaşımı ile ele alınmıştır.

**Yöntem 1**, sıfırdan geliştirilen CNN mimarisini geleneksel sliding window ve image pyramid yaklaşımıyla birleştirmektedir.

**Yöntem 2** ise önceden eğitilmiş ResNet18 backbone'unu Fully Convolutional bir detection head ile birleştirerek grid tabanlı nesne tespiti gerçekleştirmektedir.

İki yöntem arasındaki en önemli fark inference yaklaşımıdır.

Yöntem 1'de görüntünün farklı ölçekleri üzerinde çok sayıda 224×224 window değerlendirilirken, Yöntem 2 görüntünün tamamını tek seferde işleyerek grid bazında tahmin gerçekleştirmektedir.

Bu nedenle **inference hızı açısından Yöntem 2 daha avantajlıdır**.

Bununla birlikte her iki yöntemde de drone'ların çok farklı boyut, şekil ve görsel özelliklere sahip olması önemli bir zorluk oluşturmaktadır.

Özellikle Yöntem 1'de sliding-window yaklaşımının ürettiği çok sayıdaki background patch, False Positive oranını artırarak mAP@50 ve F1 performansını sınırlandırmıştır.

---

# Proje Yapısı

**`method1`** klasörü içerisinde `Yöntem 1` için olan kodlar ve sonuçlar bulunmaktadır.

**`method2`** klasörü içerisinde `Yöntem 2` için olan kodlar ve sonuçlar bulunmaktadır.

> Projeye dair daha ayrıntılı bilgi edinmek için **`Rapor.pdf`** dosyasında bulunan proje raporuna bakabilirsiniz. 

---

# Kullanılan Teknolojiler

* Python
* PyTorch
* Torchvision
* TorchMetrics
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* Pandas

---

# Temel Kavramlar

Projede aşağıdaki bilgisayarlı görü ve derin öğrenme tekniklerinden yararlanılmıştır:

* Convolutional Neural Networks
* Transfer Learning
* ResNet18
* Fully Convolutional Networks
* Sliding Window Detection
* Image Pyramid
* Grid-based Object Detection
* Bounding Box Regression
* Focal Loss
* Binary Cross Entropy
* L1 / MSE / Smooth L1 Loss
* Non-Maximum Suppression
* IoU
* Precision / Recall / F1
* mAP@50
* Hyperparameter Optimization
* Fine-Tuning

---

# Rapor

Projenin deneysel sonuçları, hiperparametre karşılaştırmaları, model mimarileri ve detaylı değerlendirmeleri **`Rapor.pdf`** sunulmuştur.

Detaylı rapor ve deney çıktıları ilgili proje dosyalarında bulunmaktadır.
