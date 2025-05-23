# Laporan Proyek Machine Learning - Saila Julia

## Domain Proyek

Penyakit sirosis hati merupakan kondisi medis kronis yang ditandai dengan kerusakan hati jangka panjang yang menyebabkan jaringan parut dan kegagalan fungsi hati. Menurut World Health Organization (WHO), penyakit hati termasuk dalam penyebab utama kematian di dunia. Deteksi dini tingkat keparahan sirosis sangat penting untuk menentukan strategi pengobatan yang efektif dan meningkatkan kualitas hidup pasien.

Namun, klasifikasi tingkat keparahan sirosis sering kali sulit dilakukan secara akurat hanya dengan penilaian klinis konvensional. Oleh karena itu, diperlukan pendekatan prediktif berbasis data untuk membantu dalam pengambilan keputusan medis

- Masalah ini penting karena keterlambatan dalam deteksi tingkat keparahan sirosis dapat menyebabkan komplikasi serius seperti gagal hati, kanker hati, dan bahkan kematian (https://aasldpubs.onlinelibrary.wiley.com/doi/pdf/10.1002/hep.29756). Dengan memanfaatkan algoritma machine learning, kita dapat membangun sistem prediksi yang memberikan informasi cepat dan akurat kepada tenaga medis (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256428)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Bagaimana membangun model prediktif akurat untuk klasifikasi
- Algoritma machine learning mana yang memberikan performa terbaik dalam hal akurasi dan kestabilan prediksi?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model klasifikasi untuk memprediksi tingkat keparahan sirosis hati dengan akurasi tinggi
- Mengevaluasi beberapa algoritma machine learning untuk menemukan model terbaik dengan kestabilan antara data latih dan uji

### Solution statements
- Menggunakan beberapa algoritma klasifikasi: Decision Tree, Random Forest, XGBoost, LightGBM, SVM, dan Logistic Regression.
- Melakukan hyperparameter tuning untuk model Random Forest, XGBoost, dan LightGBM
- Mengevaluasi model dengan metrik: akurasi, precision, recall, dan f1-score untuk memilih model terbaik berdasarkan kestabilan dan performa prediksi

## Data Understanding
Dataset yang digunakan dalam proyek.
(https://www.kaggle.com/datasets/harshitstark/prediction-of-cirrhosis-outcomes/data). 

### Variabel-variabel pada Dataset:
- age : Usia pasien
- ID : nama pengenal unik
- Sex : Jenis kelamin
- Bilirubin : tingkat birilubin dalam darah
- Albumin : Kadar Albumin
- Prothrombin : Waktu pembekuan darah
- Stage : tingkat keparahan sirosis hati
- Drug : Jenis pengobatan
- N_days : jumlah pengamatan
- Status :label target
- Ascites : penumpukan cairan pada tubuh
- Hepatomology : pencegahan
- Spiders : garis pembuluh darah
- Edema : pembengkakan
- Cholesterol : kolesterol
- Alk_Phos : tingkat kesehatan hati
- SGOT : Enzim dalam tubuh
- Tryglicerides :jumlah lemak
- Platelets : Kepingan darah
- copper : kadar mineral

### Kondisi dataset :
- Pengecekan missing values : Dataset tidak memiliki missing value dari 7905 jumlah data
- Data duplikat : tidak menemukan data duplikat
- pengecekan Outlier : Terdapat beberapa outlier pada beberapa variabel namun masi dalam rencang yang wajar
-  Distribusi Data: Data kategorikal cukup seimbang namun, ada beberapa variabel kategorikal memiliki distribusi tidak merata, seperti SEX yang didominasi perempuan
-  Tipe data : Semua kolom memiliki tipe data yang sesuai, Variabel numerik bertipe float64/int64, Variabel kategorikal bertipe object

### Ekplorasi data

1. Korelasi matrix
- korelasi positif :
  Bilirubin dengan Copper (0.44)
  Albumin dengan N_Days (0.26)
  SGOT dengan Bilirubin (0.37) dan Cholesterol (0.33)
  Prothrombin dengan Stage (0.25)
- korelasi negatif :
  Albumin dengan Stage  (-0.23)
  Albumin dengan Bilirubin (-0.30) 
  N_Days dengan Bilirubin (-0.35)
  
2. Histogram
Histogram ini menampilkan distribusi masing-masing fitur numerik dalam dataset :
-Beberapa fitur memiliki distribusi skewed ke kanan seperti Bilirubin, Cholesterol, Alk_phos, SGOT, Copper yang menunjukan banyak nilai kecil dan sedikit nilai yang sangat tinggi.
-Beberapa fitur seperti Albumin, Platelets, dan age memiliki distribusi yang relatif normal 
-Fitur stage menunjukan data dalam kategori 1-4 yang menggambarkan stadium sirosis

3. Distribusi label 
-Dataset tidak seimbang (imbalance class problem), di mana kelas CL sangat sedikit. Hal ini penting untuk diperhatikan karena model dapat bias terhadap kelas mayoritas (C dan D).
-Diperlukan teknik penanganan imbalance Oversampling (SMOTE)
-Kategori C (mungkin "Cirrhosis") merupakan kelompok terbesar, diikuti oleh D (mungkin "Death") dan CL (mungkin "Clinically Latent") sebagai yang paling sedikit

4. Distribusi jenis kelamin
- Mayoritas pasien adalah perempuan (F), yaitu sebesar 92.8% dan Pasien laki-laki (M) hanya sebesar 7.2%.
- Terdapat ketidakseimbangan gender yang cukup besar, yang bisa memengaruhi hasil model prediksi apabila variabel jenis kelamin memiliki pengaruh terhadap status sirosis.


## Data Preparation
1. Feature Engineering :
Terdapat fitur baru yang dapat diturunkan dari fitur-fitur penting dalam dataset, yaitu bilirubin, albumin, platelet, dan stage. Fitur-fitur ini memiliki peran signifikan dalam mengevaluasi tingkat keparahan sirosis hati dan membantu dalam prediksi outcome klinis pasien Bilirubin_Albumin_Ratio Rasio ini menunjukkan keseimbangan antara fungsi ekskresi (bilirubin) dan sintesis hati (albumin). Nilai tinggi menandakan kerusakan hati yang lebih parah. Stage_4 Fitur biner untuk mengidentifikasi pasien dengan sirosis tahap lanjut (Stage 4), yang memiliki risiko komplikasi lebih besar. Albumin_x_Platelets Kombinasi albumin dan trombosit menggambarkan kondisi sintesis hati dan sirkulasi darah. Nilai rendah bisa menunjukkan sirosis berat atau hipertensi portal
2. Splitting data :
- memisahkan fitur data dan label
- Memisahkan dataset menjadi data latih (80%) dan data uji (20%) menggunakan modul train_test_split dari library scikit-learn
3. Standarisasi dan Encoding :
- Pemilihan kolom : mendeteksi kolom numerical dan kategorical otomatis dari data latih
- Preprocessing Data : Mengubah variabel kategorikal menjadi representasi numerik menggunakan One-Hot Encoding dan mengubah skala fitur numerik menjadi skala seragam menggunakan StandardScaler.
4. Pipeline :
- Penggunaan SMOTE : sebagai penanganan imbalance data
- Preprocessor sebagai fungsi melakukan proses standarisasi dan encoding digabungkan sebagai Pipeline dari library scikit-learn
5. Transformasi data : Menerapkan preprocessing (standarisasi dan encoding)  pada data latih dan data uji untuk mempersiapkan data bagi model
6. Tunning : Agar model lebih akurat 

## Modeling
1. Decision Tree
Kelebihan dari Decision Tree terletak pada kemampuannya yang mudah dijelaskan serta mendukung baik data numerik maupun kategorikal. Namun, metode ini rentan terhadap overfitting, terutama jika struktur pohon terlalu dalam. Selain itu, Decision Tree cenderung sensitif terhadap ketidakseimbangan kelas dan bisa menunjukkan performa yang tidak konsisten.

- Prinsip kerja:
Memulai dengan membagi data berdasarkan fitur paling informatif.
Penentuan titik split menggunakan metrik seperti Gini Index atau Entropy.
Proses pemisahan dilakukan secara berulang hingga membentuk cabang pohon.
Setiap node akhir (leaf) mewakili klasifikasi berdasarkan mayoritas data.

- Langkah implementasi:
Model diinisialisasi dengan random_state=2024.
Pelatihan dilakukan pada data training yang telah dipersiapkan.
Prediksi dilakukan untuk data pelatihan dan pengujian.
Evaluasi performa memakai skor rata-rata berbobot (weighted).

- Parameter penting:
random_state=2024 untuk memastikan hasil yang konsisten.
Parameter lainnya dibiarkan default.
average="weighted" digunakan agar hasil tidak terpengaruh ketimpangan jumlah kelas.

2. Random Forest
Random Forest menawarkan kestabilan yang baik dan cukup tahan terhadap noise dalam data. Kekurangannya termasuk komputasi yang berat, interpretasi yang tidak sederhana, serta memerlukan jumlah data yang memadai.

- Cara kerja:
Membuat beberapa pohon keputusan secara paralel.
Setiap pohon dilatih dari sampel acak (bagging).
Hanya subset fitur yang digunakan pada tiap pemisahan.
Hasil akhir ditentukan lewat voting mayoritas

- Langkah implementasi:
Gunakan RandomForestClassifier dengan 100 pohon (n_estimators=100).
Latih model pada data pelatihan.
Lakukan prediksi menggunakan metode ensemble.
Evaluasi dengan skor rata-rata berbobot.

- Parameter utama:
n_estimators=100.
Kriteria pemisahan memakai nilai default.
Evaluasi memakai metode average="weighted" untuk kompensasi kelas tidak seimbang.

3. XGBoost
Kelebihan utama XGBoost adalah kecepatannya, efisiensi komputasi, dan kemampuannya mengurangi risiko overfitting. Meski begitu, model ini relatif kompleks, membutuhkan sumber daya besar, dan perlu penyesuaian parameter yang akurat agar optimal

- Mekanisme kerja:
Model dibangun secara bertahap (boosting).
Setiap iterasi fokus memperbaiki kesalahan dari prediksi sebelumnya.
Optimasi dilakukan menggunakan pendekatan penurunan gradien.
Gabungan semua model lemah membentuk prediksi akhir.

- Langkah-langkah:
Gunakan GradientBoostingClassifier sebagai inisialisasi model.
Latih model menggunakan dataset training.
Lakukan prediksi secara bertahap.
Evaluasi menggunakan skor rata-rata berbobot

- Parameter yang digunakan:
Learning rate dan jumlah estimator mengikuti nilai default.
Evaluasi hasil menggunakan average="weighted"

4. SVM :
SVM sangat efektif ketika menangani data berdimensi tinggi. Akan tetapi, metode ini memerlukan komputasi yang intensif, sulit ditafsirkan secara intuitif, dan perlu pemilihan kernel yang sesuai agar efektif.

- Konsep kerja:
Menentukan hyperplane terbaik untuk memisahkan kelas.
Menerapkan kernel trick untuk mengatasi data non-linear.
Fokus memaksimalkan margin antar kelas.
Mengandalkan support vectors untuk proses klasifikasi.

- Tahapan penerapan:
Inisialisasi model dengan SVC(kernel='rbf').
Latih model pada data training yang sudah distandarisasi.
Lakukan prediksi terhadap data baru.
Gunakan evaluasi berbobot untuk menilai performa.

- Parameter:
Kernel RBF digunakan untuk data non-linear.
random_state=2024 dipakai untuk konsistensi.
Evaluasi melalui average="weighted"

5. LightGBM
LightGBM dikenal cepat dan efisien, dengan performa tinggi terutama pada dataset besar. Di sisi lain, model ini butuh banyak sumber daya, memerlukan penyesuaian parameter, dan interpretasinya tidak mudah

- Prinsip kerja:
Menerapkan metode boosting berbasis gradien.
Menggunakan pendekatan leaf-wise untuk pemisahan optimal.
Mengadopsi histogram learning untuk efisiensi memori.
Memungkinkan pelatihan secara paralel.

- Langkah-langkah implementasi:
Inisialisasi model dengan LGBMClassifier.
Lakukan pelatihan pada data yang telah diproses.
Prediksi dilakukan secara bertahap.
Evaluasi menggunakan rata-rata berbobot

- Parameter:
Parameter default dipertahankan untuk learning rate dan jumlah daun.
Digunakan average="weighted" untuk penyesuaian terhadap distribusi kelas yang tidak merata.

6. Logistic Regression
Logistic Regression dikenal karena sederhana dan mudah dipahami. Meskipun begitu, algoritma ini tidak ideal untuk masalah kompleks, sensitif terhadap pencilan, dan kinerjanya bisa terganggu jika terdapat ketidakseimbangan kelas

- Cara kerja:
Menghitung probabilitas tiap kelas menggunakan fungsi logistik.
Untuk klasifikasi multi-kelas, digunakan pendekatan multinomial.
Proses optimasi dilakukan melalui gradient descent.
Model menghasilkan probabilitas, bukan hanya label kelas

- Langkah-langkah:
Gunakan LogisticRegression(multi_class='multinomial', max_iter=1000).
Lakukan pelatihan model pada data training.
Prediksi hasil dalam bentuk probabilitas.
Evaluasi hasil prediksi dengan skor berbobot.

- Parameter:
multi_class='multinomial' untuk multi-klasifikasi.
max_iter=1000 agar proses konvergen.
Evaluasi menggunakan average="weighted"

## Evaluation
Metrik yang digunakan
Dalam proyek klasifikasi ini, evaluasi performa model dilakukan dengan menggunakan empat metrik utama, yaitu akurasi, presisi, recall, dan F1-score. Setiap metrik memberikan perspektif berbeda terhadap kinerja model, terutama dalam konteks data yang memiliki distribusi kelas tidak seimbang.

1. Akurasi
Akurasi menggambarkan sejauh mana prediksi model sesuai dengan label sebenarnya. Metrik ini dihitung sebagai rasio antara jumlah prediksi yang benar terhadap seluruh jumlah data.
Rumus:
Akurasi = (Prediksi Benar) / (Total Data)
Meskipun sederhana, akurasi bisa menyesatkan jika kelas tidak terdistribusi secara merata. Oleh karena itu, perlu dilengkapi dengan metrik lain.

2. Presisi
Presisi menunjukkan proporsi prediksi positif yang benar-benar tepat. Ini penting terutama saat biaya dari kesalahan prediksi positif tinggi.
Rumus:
Presisi = True Positive / (True Positive + False Positive)
Presisi digunakan untuk mengetahui seberapa andal model dalam memberikan hasil positif yang akurat.

3.  Recall
Recall mengukur kemampuan model dalam mengenali seluruh kasus positif yang ada. Metrik ini berguna saat kegagalan mendeteksi kasus positif dianggap kritis.
Rumus:
Recall = True Positive / (True Positive + False Negative)
Semakin tinggi nilai recall, semakin baik model dalam menangkap seluruh kasus yang seharusnya diklasifikasikan positif.

4. F1-Score
F1-Score merupakan rata-rata harmonis dari presisi dan recall. Metrik ini sangat berguna ketika diperlukan keseimbangan antara dua metrik tersebut, terutama dalam kondisi distribusi kelas yang tidak seimbang.
Rumus:
F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)

klasifikasi umum berdasarkan skor F1:
Tinggi (0.90–1.00): Model sangat presisi dan sensitif
Sedang (0.70–0.89): Model cukup baik dan seimbang
Rendah (0.50–0.69): Model kurang optimal
Sangat rendah (<0.50): Model gagal menangani klasifikasi dengan baik

### Penerapan Metrik pada Evaluasi Model
- Decision Tree:
Memiliki akurasi dan F1-score sangat tinggi pada data latih (100%), namun anjlok di data uji (72%). Hal ini menandakan overfitting. Selain itu, presisi dan recall tidak seimbang pada kelas CL, menghasilkan performa minoritas yang sangat buruk.

- XGBoost:
Mencapai akurasi 86% di pelatihan dan 82% di pengujian, menunjukkan stabilitas performa yang baik. Presisi dan recall seimbang di seluruh kelas, menghasilkan F1-score tinggi yang menandakan model sangat andal dan akurat.

- Random Forest:
Hasil training sempurna (100%), sementara testing stabil di 82%. Namun, seperti Decision Tree, model ini kesulitan mendeteksi kelas minoritas (CL), dengan recall sangat rendah. Tetap memiliki F1-score keseluruhan yang baik.

- SVM:
Memiliki akurasi dan F1-score 86% (training) dan 82% (testing), dengan perbedaan kecil antar keduanya. Menunjukkan model tidak overfit, namun performa masih terbatas untuk kelas CL.

- Logistic Regression:
Akurasi dan F1-score berada di kisaran sedang (80% training, 77% testing). Model gagal total mendeteksi kelas CL (semua metrik = 0). Ini menunjukkan bahwa pendekatan linier tidak cukup kompleks untuk data ini.

- LightGBM:
Memberikan performa tertinggi di data uji (93%), namun dengan indikasi overfitting ringan karena hasil training yang lebih tinggi (97%). Precision dan recall kurang optimal pada kelas minoritas, tetapi F1-score masih masuk dalam kategori tinggi

### Kesimpulan Evaluasi Model
- Model Terbaik:
Berdasarkan kombinasi semua metrik, XGBoost, Random Forest, dan LightGBM memberikan hasil terbaik. Ketiganya mencatat skor tinggi pada akurasi, precision, recall, dan F1-score saat testing.

- Model Paling Stabil:
XGBoost menunjukkan stabilitas performa terbaik antara data latih dan uji, tanpa indikasi overfitting serius.

- Performa Tertinggi:
LightGBM mencapai skor tertinggi pada data uji (93%) meskipun dengan indikasi overfitting ringan.

- Overfitting:
Terjadi paling parah pada Decision Tree, disusul oleh Random Forest dan LightGBM. Sementara SVM dan Logistic Regression menunjukkan performa lebih stabil, tetapi dengan hasil metrik yang lebih rendah.

## Kesimpulan 
Dalam pengembangan model prediksi untuk klasifikasi sirosis hati, beberapa tantangan utama berhasil diatasi:

1. Penyusunan Model Prediktif yang Efektif
   Model yang dikembangkan menunjukkan performa yang baik berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score. Meski beberapa model belum optimal dalam mengenali kelas minoritas (khususnya kelas CL), secara keseluruhan model cukup andal dalam mengklasifikasikan kategori sirosis yang lebih umum. Ini membuktikan bahwa solusi yang dibangun mampu memenuhi kebutuhan akan model prediksi yang akurat.
   
3. Pencapaian Target Akurasi 85%
   Beberapa model berhasil mencapai dan bahkan melampaui target akurasi minimum sebesar 85%. LightGBM mencatat akurasi tertinggi pada data uji sebesar 93%, sementara XGBoost dan Random Forest masing-masing mencapai 82%, yang menunjukkan kinerja yang kuat dan konsisten. Hasil ini membuktikan bahwa tujuan utama proyek telah berhasil dicapai.
   
4. Evaluasi Komprehensif Terhadap Beragam Algoritma
   Enam model machine learning Decision Tree, XGBoost, Random Forest, SVM, Logistic Regression, dan LightGBM telah dianalisis secara menyeluruh. Evaluasi berdasarkan metrik yang relevan menunjukkan bahwa XGBoost memberikan performa paling seimbang dan stabil antara data latih dan uji, dengan minim overfitting. Di sisi lain, LightGBM memiliki akurasi tertinggi, namun dengan indikasi overfitting ringan, sedangkan Random Forest menunjukkan performa yang baik namun cenderung overfit pada data pelatihan.

### Rekomendasi
Berdasarkan hasil evaluasi, **XGBoost** merupakan pilihan model yang paling direkomendasikan untuk digunakan dalam sistem prediksi obesitas. Model ini menunjukkan kombinasi terbaik antara akurasi tinggi, stabilitas performa, serta risiko overfitting yang rendah dibandingkan model lainnya.
