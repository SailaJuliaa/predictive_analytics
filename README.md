# Laporan Proyek Machine Learning - Saila Julia

## Domain Proyek

Penyakit sirosis hati merupakan kondisi medis kronis yang ditandai dengan kerusakan hati jangka panjang yang menyebabkan jaringan parut dan kegagalan fungsi hati. Menurut World Health Organization (WHO), penyakit hati termasuk dalam penyebab utama kematian di dunia. Deteksi dini tingkat keparahan sirosis sangat penting untuk menentukan strategi pengobatan yang efektif dan meningkatkan kualitas hidup pasien.

Klasifikasi tingkat keparahan sirosis hati menjadi sangat penting untuk menentukan tindakan medis yang tepat serta memprediksi prognosis pasien. Namun, proses klasifikasi ini seringkali memerlukan prosedur medis yang mahal, invasif, dan tidak selalu tersedia secara merata, khususnya di daerah dengan sumber daya kesehatan terbatas. Maka, diperlukan pendekatan prediktif berbasis data untuk membantu dalam pengambilan keputusan medis

- Masalah ini penting karena keterlambatan dalam memprediksi outcome sirosis hati dapat menyebabkan luputnya penanganan dini terhadap risiko komplikasi serius seperti gagal hati, kanker hati, dan bahkan kematian. Dengan memanfaatkan algoritma machine learning, kita dapat membangun sistem prediksi yang mampu memberikan informasi cepat dan akurat kepada tenaga medis mengenai potensi hasil klinis pasien, sehingga memungkinkan intervensi yang lebih tepat waktu dan berbasis data.

 ([https://pmc.ncbi.nlm.nih.gov/articles/PMC11867977/]) ([https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-024-05726-2])

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Apakah data klinis seperti usia, hasil laboratorium, dan kondisi medis terkait dapat digunakan untuk memprediksi outcome pasien sirosis hati secara akurat?
- Dapatkah model prediksi berbasis machine learning membantu tenaga medis dalam pengambilan keputusan yang lebih cepat dan akurat untuk pasien sirosis hati?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model prediksi berbasis machine learning yang dapat mengklasifikasikan outcome pasien sirosis hati ke dalam beberapa kategori
- Mengevaluasi beberapa algoritma machine learning untuk menemukan model terbaik dengan kestabilan antara data latih dan uji

### Solution statements
- Solusi yang ditawarkan adalah pengembangan sistem prediksi outcomes pasien sirosis hati dengan memanfaatkan algoritma klasifikasi machine learning : Decision Tree, Random Forest, XGBoost, LightGBM, SVM, dan Logistic Regression.
- Penerapan teknik feature engineering untuk meningkatkan kualitas fitur prediktif.
- Mengevaluasi model dengan metrik: akurasi, precision, recall, dan f1-score untuk memilih model terbaik berdasarkan kestabilan dan performa prediksi

## Data Understanding
Dataset yang digunakan dalam proyek.
(https://www.kaggle.com/datasets/harshitstark/prediction-of-cirrhosis-outcomes/data). 

Jumlah data awal pada dataset:
- Jumlah baris : 7905
- Jumlah kolom : 20

### Kolom pada Dataset mencangkup:
- Id : Pengenal Unik (unique id)
- age : Usia pasien.
- Sex : Jenis kelamin sering memengaruhi prognosis dan respons pengobatan.
- Bilirubin : kadar bilirubin Indikator utama fungsi hati.
- Albumin : Menunjukkan status nutrisi dan fungsi hati.
- Prothrombin : Indikator fungsi pembekuan darah dan fungsi hati.
- Stage : Tingkat keparahan sirosis.
- Drug : Jenis pengobatan yang diterima
- Status : Label target
- Ascites : Gejala klinis Penumpukan cairan di perut.
- Hepatomology : pencegahan atau riwayat penyakit hati.
- Spiders : Pembuluh darah kecil di kulit, Tanda klinis sirosis
- Edema : Pembengkakan.
- Cholesterol : Jumlah kolesterol yang berhubungan dengan metabolisme hati.
- Alk_Phos (Alkaline Phosphatase) : Enzim yang mengindikasikan kesehatan hati.
- SGOT : Enzim hati AST.
- Tryglicerides : Kadar lemak dalam darah.
- Platelets : Jumlah keping pada darah.
- copper : Kadar tembaga untuk mendeteksi kelainan seperti wilson.
- N_days : jumlah pengamatan

### Kondisi dataset :
- Pengecekan missing values : Dataset tidak memiliki missing value dari 7905 baris data 
- Data duplikat : tidak menemukan data duplikat
- pengecekan Outlier : terdapat extreme outlier pada numeric kolom yaitu birilubin, cholesterol, SGOT, Alk_Phos, Tryglicerides


### Ekplorasi data
#### Univariate analysis
1. Numerical feature. 
- Bilirubin, Cholesterol, Copper, SGOT, Alk_Phos, Triglycerides menunjukkan distribusi right-skewed (banyak nilai ekstrem tinggi).
- Albumin, Prothrombin, Platelets cenderung normal atau mendekati simetris.
- Stage penyakit menunjukkan distribusi data yang hampir merata antar tahap (1–4), meskipun Stage 2 dan Stage 4 sedikit lebih dominan.
- N_Days memiliki distribusi skewed, dengan sebagian besar pasien berada di bawah 2000 hari.
  
![univariate numerical](https://github.com/user-attachments/assets/555c6981-aad9-48fe-9115-38dae7d49c60)

2. Categorical Feature
#### Status:
 - Kategori C mendominasi dengan jumlah > 4500.
 - D sekitar 2200-an.
 - CL sangat kecil (< 500), memperkuat indikasi class imbalance.
   
![Univariate categorical1](https://github.com/user-attachments/assets/7a9ef2c9-5d68-403e-befd-65e8f1e791bc)

#### Jenis Kelamin (Sex):
 - Dominasi besar pada pasien perempuan (F) > 6500.
 - Laki-laki (M) sangat sedikit (< 1000)
 - Ini menunjukkan ketimpangan distribusi gender yang bisa menjadi pertimbangan dalam interpretasi hasil model.
   
![Univariate categorical1](https://github.com/user-attachments/assets/7a9ef2c9-5d68-403e-befd-65e8f1e791bc)

#### Multivariate Analysis
1. Categorical Feature  'Status'
Terdapat tiga kategori status pasien: C, D, dan CL.
- Status C (Compensated Cirrhosis) mendominasi jumlah pasien.
- Status D (Decompensated Cirrhosis) berada di posisi kedua.
- Status CL (Cirrhosis Liver Failure) memiliki jumlah paling sedikit.
- Distribusi yang tidak seimbang ini mengindikasikan adanya potensi class imbalance yang perlu ditangani dalam proses pemodelan prediktif.

![distribusi kategorikal](https://github.com/user-attachments/assets/b53dbac2-4cc3-4ca7-adb9-e133ac4da1f6)

2. Numerical Feature
Korelasi antar Fitur Numerik

Korelasi tertinggi terjadi antara:
   - SGOT dan Alk_Phos (r = 0.43)
   - Alk_Phos dan Cholesterol (r = 0.35)
   - SGOT dan Bilirubin (r = 0.35)
     
Korelasi negatif ditemukan antara Albumin dengan Bilirubin dan SGOT, yang berpotensi sebagai indikator kerusakan hati.
 - Tidak ditemukan multikolinearitas ekstrem antar fitur (tidak ada nilai korelasi > 0.8), sehingga tidak perlu melakukan penghapusan fitur karena kolinearitas.

 ![korelasi matrix](https://github.com/user-attachments/assets/496bbe7a-b72f-47a3-b62e-2f1172f22cbd)

#### Kesimpulan pada Ekplorasi Data
- Distribusi Fitur Kategorikal bersifat tidak seimbang, terutama pada fitur Sex dan Status.
- Fitur numerik menunjukkan berbagai bentuk distribusi, banyak yang tidak normal dan skewed.
- Korelasi antar fitur numerik lemah sampai sedang, tidak ada multikolinearitas tinggi.


## Data Preparation
#### 1. Hapus Outlier :
- Menghapus outlier extreme pada kolom birilubin, cholesterol, SGOT, Alk_Phos, Trygliceridesdari
- Jumlah baris dan kolom hasil dari penghapusan outlier adalah : 7089 baris dan 20 kolom dari data awal 7905 baris dan 20 kolom
- Mengganti nama dataframe hasil outlier yaitu train_cleaned dan test_cleaned menjadi train dan test
  
#### 2. Feature Engineering :
- Rasio Bilirubin terhadap Albumin untuk membuat fitur baru yang mencerminkan keseimbangan antara dua indikator penting fungsi hati: Bilirubin (indikator kerusakan hati) dan Albumin (indikator fungsi hati).
- Fitur Biner untuk stadium 4 (Stage 4) untuk membuat fitur biner yang merepresentasikan apakah pasien berada di tahap paling parah (Stage 4) dari penyakit.
- Interaksi antara Albumin dan Platelet untuk menambahkan fitur interaksi antara dua variabel numerik penting.

##### 3. Encoding Target Variabel 'Status':
- Menggunakan labelEncoder untuk tahap encoding
- Tujuan encoding variabel status untuk mengubah target kategorikal (Status: C, D, CL) menjadi format numerik agar bisa digunakan dalam algoritma pembelajaran mesin, khususnya XGBoost.

#### 4. Splitting Data:
- Pemisahan fitur dan target untuk memisahkan input (fitur X) dan output (target y) untuk keperluan pelatihan model agar model hanya belajar dari fitur bukan target
  
#### 5. Train_test split :
- Membagi data menjadi 80% data latih dan 20% data uji, dengan mempertahankan proporsi kelas target menggunakan stratify
- Tujuan stratifikasi untuk mencegah ketimpangan distribusi kelas di data latih dan uji, yang penting dalam kasus class imbalance seperti ini

#### 6. Preprocessing:
- Identifikasi fitur numeric dan kategorical dengan tujuan untuk Memisahkan fitur berdasarkan jenis datanya untuk preprocessing yang sesuai
- Standarisasi dan One-Hot Encoding menggunakan *StandardScaler* untuk menstandarkan fitur numerik menjadi distribusi dengan mean 0 dan std dev 1, dan *OneHotEncoder* untuk mengubah fitur kategorikal menjadi format biner
- Keuntungan : Standardisasi membantu algoritma seperti SVM, Logistic Regression, dan XGBoost agar tidak bias terhadap fitur berskala besar. One-hot encoding membuat data kategorikal bisa dibaca oleh model sambil menghindari dummy variable trap (drop='first').

#### 7. Pembuatan Processor
-  Menerapkan transformasi yang berbeda ke fitur numerik dan kategorikal dalam satu objek pipeline menggunakan _ColumnTransformer_.
-  Keuntungan nya efisien dan modular, mudah diintegrasikan ke dalam pipeline pemodelan.
Meningkatkan keterbacaan kode

## Modeling
#### 1. Decision Tree
Kelebihan dari Decision Tree terletak pada kemampuannya yang mudah dijelaskan serta mendukung baik data numerik maupun kategorikal. Namun, metode ini rentan terhadap overfitting, terutama jika struktur pohon terlalu dalam. Selain itu, Decision Tree cenderung sensitif terhadap ketidakseimbangan kelas dan bisa menunjukkan performa yang tidak konsisten.

- Prinsip kerja:
Membagi data berdasarkan fitur paling informatif menggunakan Gini atau Entropy.
Pemisahan dilakukan rekursif hingga semua data terklasifikasi.t.

- Langkah implementasi:
_random_state_ tidak diset secara eksplisit, menggunakan default.
Evaluasi menggunakan skor rata-rata berbobot
Pelatihan dilakukan pada data training yang telah dipersiapkan.
Prediksi dilakukan untuk data pelatihan dan pengujian.

- Parameter penting:
_random_state_ menggunakan default.


#### 2. Random Forest
Random Forest menawarkan kestabilan yang baik dan cukup tahan terhadap noise dalam data. Kekurangannya termasuk komputasi yang berat, interpretasi yang tidak sederhana, serta memerlukan jumlah data yang memadai.

- Cara kerja:
Membuat beberapa pohon keputusan secara paralel.
Setiap pohon dilatih dari sampel acak (bagging).
Hanya subset fitur yang digunakan pada tiap pemisahan.
Hasil akhir ditentukan lewat voting mayoritas

- Langkah implementasi:
Gunakan _RandomForestClassifier_ menggunakan default (n_estimators=100).
Latih model pada data pelatihan.
Evaluasi menggunakan _average='weighted'_

- Parameter utama:
n_estimators=100.
Kriteria pemisahan memakai nilai default.
Evaluasi memakai metode average="weighted" untuk kompensasi kelas tidak seimbang.

#### 3. XGBoost
Kelebihan utama XGBoost adalah kecepatannya, efisiensi komputasi, dan kemampuannya mengurangi risiko overfitting. Meski begitu, model ini relatif kompleks, membutuhkan sumber daya besar, dan perlu penyesuaian parameter yang akurat agar optimal

- Mekanisme kerja:
Model dibangun secara bertahap (boosting).
Setiap iterasi fokus memperbaiki kesalahan dari prediksi sebelumnya.
Optimasi dilakukan menggunakan pendekatan penurunan gradien.
Gabungan semua model lemah membentuk prediksi akhir.

- Langkah-langkah:
 Menggunakan _use_label_encoder=False_ untuk Mencegah peringatan.
Latih model menggunakan dataset training.
Lakukan prediksi secara bertahap.
_eval_metric='logloss'_ Untuk evaluasi klasifikasi
Evaluasi _average='weighted'_

- Parameter yang digunakan:
Evaluasi hasil menggunakan average="weighted"

#### 4. SVM :
SVM sangat efektif ketika menangani data berdimensi tinggi. Akan tetapi, metode ini memerlukan komputasi yang intensif, sulit ditafsirkan secara intuitif, dan perlu pemilihan kernel yang sesuai agar efektif.

- Konsep kerja:
Menentukan hyperplane terbaik untuk memisahkan kelas.
Menerapkan kernel trick untuk mengatasi data non-linear.
Fokus memaksimalkan margin antar kelas.
Mengandalkan support vectors untuk proses klasifikasi dan kernel RBF.

- Tahapan penerapan:
Inisialisasi model dengan _SVC()_.
Latih model pada data training yang sudah distandarisasi.
Lakukan prediksi terhadap data baru.
Gunakan evaluasi berbobot untuk menilai performa.

- Parameter:
Default kernel adalah _rbf_, cocok untuk data non-linear.
Evaluasi menggunakan _average='weighted'_

#### 5. LightGBM
LightGBM dikenal cepat dan efisien, dengan performa tinggi terutama pada dataset besar. Di sisi lain, model ini butuh banyak sumber daya, memerlukan penyesuaian parameter, dan interpretasinya tidak mudah

- Prinsip kerja:
Menerapkan metode boosting berbasis gradien.
Menggunakan pendekatan leaf-wise untuk pemisahan optimal.
Mengadopsi histogram learning untuk efisiensi memori.
Memungkinkan pelatihan secara paralel.

- Langkah-langkah implementasi:
Inisialisasi model dengan _LGBMClassifier()_.
Lakukan pelatihan pada data yang telah diproses.
Prediksi dilakukan secara bertahap.
Evaluasi menggunakan rata-rata berbobot

- Parameter:
Parameter default dipertahankan untuk learning rate dan jumlah daun.
Digunakan average="weighted" untuk penyesuaian terhadap distribusi kelas yang tidak merata.

#### 6. Logistic Regression
Logistic Regression dikenal karena sederhana dan mudah dipahami. Meskipun begitu, algoritma ini tidak ideal untuk masalah kompleks, sensitif terhadap pencilan, dan kinerjanya bisa terganggu jika terdapat ketidakseimbangan kelas

- Cara kerja:
Menghitung probabilitas tiap kelas menggunakan fungsi logistik.
Untuk klasifikasi multi-kelas, digunakan pendekatan multinomial.
Proses optimasi dilakukan melalui gradient descent.
Model menghasilkan probabilitas, bukan hanya label kelas

- Langkah-langkah:
Gunakan _LogisticRegression(max_iter=1000)_
Lakukan pelatihan model pada data training.
Prediksi hasil dalam bentuk probabilitas.
Evaluasi hasil prediksi dengan skor berbobot.

- Parameter:
_max_iter=1000_ Meningkatkan jumlah iterasi agar proses konvergen
Multi-klasifikasi otomatis didukung 
Evaluasi menggunakan _average="weighted"_

## Evaluation
Metrik yang digunakan
Dalam proyek klasifikasi ini, evaluasi performa model dilakukan dengan menggunakan empat metrik utama, yaitu akurasi, presisi, recall, dan F1-score. Setiap metrik memberikan perspektif berbeda terhadap kinerja model, terutama dalam konteks data yang memiliki distribusi kelas tidak seimbang.

### 1. Akurasi

Akurasi menggambarkan sejauh mana prediksi model sesuai dengan label sebenarnya. Metrik ini dihitung sebagai rasio antara jumlah prediksi yang benar terhadap seluruh jumlah data.
Rumus:
Akurasi = (Prediksi Benar) / (Total Data)
Meskipun akurasi mudah dipahami, metrik ini bisa menyesatkan jika distribusi kelas tidak seimbang, seperti pada kasus ini (jumlah data kelas CL sangat sedikit).
- Logistic Regression	0.790
- Decision Tree	0.746
- Random Forest	0.819
- SVM	0.808
- XGBoost	0.820
- LightGBM	0.827
  
_XGBoost menunjukkan akurasi tertinggi, disusul oleh LightGBM dan Random Forest_

### 2. Presisi

Presisi menunjukkan proporsi prediksi positif yang benar-benar tepat. Ini penting terutama saat biaya dari kesalahan prediksi positif tinggi.
Rumus:
Presisi = True Positive / (True Positive + False Positive)
Presisi digunakan untuk mengetahui seberapa andal model dalam memberikan hasil positif yang akurat.
- Logistic Regression	0.759
- Decision Tree	0.739
- Random Forest	0.821
- SVM	0.776
- XGBoost	0.809
- LightGBM	0.817
  
_LightGBM dan Random Forest memiliki presisi tertinggi, menunjukkan model cenderung akurat saat memberikan prediksi positif_

### 3.Recall
Recall mengukur kemampuan model dalam mengenali seluruh kasus positif yang ada. Metrik ini berguna saat kegagalan mendeteksi kasus positif dianggap kritis.
Rumus:
Recall = True Positive / (True Positive + False Negative)
Semakin tinggi nilai recall, semakin baik model dalam menangkap seluruh kasus yang seharusnya diklasifikasikan positif.
- Logistic Regression	0.790
- Decision Tree	0.746
- Random Forest	0.819
- SVM	0.808
- XGBoost	0.820
- LightGBM	0.827
  
_XGBoost dan LightGBM kembali menunjukkan performa tinggi, mengindikasikan kemampuan dalam mendeteksi seluruh kelas_

### 4.F1-Score
F1-Score merupakan rata-rata harmonis dari presisi dan recall. Metrik ini sangat berguna ketika diperlukan keseimbangan antara dua metrik tersebut, terutama dalam kondisi distribusi kelas yang tidak seimbang.
Rumus:
F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)
- Logistic Regression	0.767
- Decision Tree	0.743
- Random Forest	0.805
- SVM	0.788
- XGBoost	0.808
- LightGBM	0.814
  
_LightGBM memiliki keseimbangan terbaik antara presisi dan recall_


## Penerapan Metrik pada Evaluasi Model
### Hasil Evaluasi
#### 1.Logistic Regression
- Training Accuracy, Precision, Recall, F1-score: 78%
- Testing Accuracy, Precision, Recall, F1-score: 79%
- Performa antara data latih dan data uji relatif konsisten, sehingga tidak ada indikasi overfitting yang signifikan.
- Precision dan recall sangat tidak seimbang antar kelas, terutama pada kelas CL yang memiliki nilai 0 untuk semua metrik, menandakan model sama sekali gagal mengenali kelas ini.
- F1-score keseluruhan tergolong sedang, yang menunjukkan bahwa model cukup baik dalam mengenali kelas mayoritas.

#### 2. Decision Tree
- Training Accuracy, Precision, Recall, F1-score: 100%
- Testing Accuracy, Precision, Recall, F1-score: 75%
- Perbedaan besar antara performa data latih dan data uji menunjukkan adanya overfitting pada model.
- Precision dan recall terlihat tidak seimbang antar kelas, terutama pada kelas CL yang memiliki performa sangat rendah.
- F1-score keseluruhan tergolong sedang, yang menunjukkan bahwa meskipun model cukup akurat untuk kelas mayoritas, performa pada kelas minoritas masih perlu diperbaiki.

#### 3. Random Forest
- Training Accuracy, Precision, Recall, F1-score: 100%
- Testing Accuracy, Precision, Recall, F1-score: 82%
- Performa model pada data latih dan uji tergolong stabil meskipun ada indikasi overfitting ringan.
- Precision dan recall memiliki kinerja yang sangat baik pada kelas mayoritas C dan D, namun pada kelas CL, performa model sangat rendah (precision 1.00, recall 0.10).
- F1-score keseluruhan tergolong cukup baik, tetapi ketidakseimbangan performa antar kelas perlu diperbaiki.

#### 4. SVM
- Training Accuracy, Precision, Recall, F1-score: 81%
- Testing Accuracy, Precision, Recall, F1-score: 81%
- Perbedaan performa antara data latih dan data uji tidak terlalu besar, menunjukan bahwa model tidak mengalami overfitting secara signifikan.
- Precision dan recall terlihat tidak seimbang pada kelas CL yang memiliki nilai 0 untuk semua metrik
- F1-score tergolong baik, yang berarti model cukup akurat untuk mengenali kelas mayoritas.

#### 5. XGBoost
- Training Accuracy, Precision, Recall, F1-score: 99%
- Testing Accuracy, Precision, Recall, F1-score: 82%
- Perbandingan training dan testing akurasi menunjukkan overfitting namun dengan performa testing yang sangat baik.
- Precision dan recall yang cukup seimbang pada kelas mayoritas, namun masih rendah untuk kelas CL.
- F1-score pada kategori tinggi yang berarti model sangat akurat secara keseluruhan.

#### 6. LightGBM
- Training Accuracy, Precision, Recall, F1-score: 96%
- Testing Accuracy, Precision, Recall, F1-score: 83%
- Perbedaan performa antara data latih dan data uji cukup besar, yang menunjukkan kemungkinan adanya overfitting ringan pada model.
- Precision dan recall antar kelas masih belum seimbang, terutama pada kelas CL yang memiliki recall sangat rendah.
- F1-score tergolong baik yang berarti model cukup akurat dalam memprediksi sebagian besar kelas.

### Kesimpulan Evaluasi Model
#### Performa terbaik:
- Model XGBoost, Random Forest, dan LightGBM memberikan performa terbaik berdasarkan metrik akurasi, precision, recall, dan f1-score.

#### Performa sangat tinggi
- XGBoost menunjukkan performa yang sangat tinggi dan cukup stabil antara data latih (99%) dan uji (82%), meskipun sedikit overfit
  
#### Akurasi tertinggi
- Random Forest mencatat akurasi tinggi pada data uji (82%) namun sangat overfit pada data latih (100%)

#### Akurasi cukup baik
- LightGBM memiliki hasil akurasi yang baik (83% pada testing), namun masih menunjukkan potensi overfitting ringan

_Ketiganya memiliki hasil testing accuracy, precision, recall, dan f1-score di kisaran 80%, yang tergolong tinggi._

#### Overfitting:
- Terjadi paling parah pada Decision Tree (100% training, 75% testing), serta Random Forest dan LightGBM, meskipun dampaknya lebih ringan. Sementara XGBoost, SVM, dan Logistic Regression menunjukkan kestabilan antara data training dan testing, dengan Logistic Regression memiliki performa paling rendah di antara model-model lain.
  
## Kesimpulan 
- Dalam pengembangan model prediksi untuk klasifikasi tingkat sirosis hati, beberapa hal penting telah dicapai dan menjadi sorotan utama dalam proses modeling:

### 1. Pembangunan Model Prediktif yang Andal
Model-model yang dikembangkan telah menunjukkan performa yang cukup baik berdasarkan metrik evaluasi utama seperti akurasi, presisi, recall, dan F1-score. Tantangan terbesar ditemukan pada kelas minoritas (CL) yang memiliki jumlah data jauh lebih sedikit dibandingkan kelas lain (C dan D). Meskipun begitu, sebagian besar model berhasil mengklasifikasikan kelas mayoritas dengan cukup akurat, yang menunjukkan bahwa sistem prediksi ini telah bekerja secara efektif dalam mengenali pola-pola umum yang lazim dijumpai di dunia nyata.
   
### 2. Capaian Akurasi Model pada Data Uji
Berdasarkan hasil evaluasi pada data uji, belum ada model yang mencapai target akurasi 85%, namun beberapa model mendekati nilai tersebut dan menunjukkan performa yang layak digunakan.
Akurasi tertinggi pada data uji dicapai oleh:
- LightGBM: 82.7%
- XGBoost: 82.0%
- Random Forest: 81.9%
Ketiga model tersebut menampilkan performa yang relatif konsisten, meskipun masih terdapat potensi overfitting ringan terutama pada LightGBM dan Random Forest (dilihat dari gap akurasi antara data latih dan uji).
Model lain seperti Decision Tree menunjukkan akurasi uji yang rendah (74.6%) dan indikasi overfitting parah (akurasi latih 99.9%), sementara SVM dan Logistic Regression lebih stabil tetapi tidak unggul dalam hal akurasi maupun deteksi kelas minoritas.

### 3. Evaluasi Menyeluruh terhadap Beragam Model
Enam algoritma telah diuji dan dievaluasi secara komprehensif, yaitu: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, dan LightGBM. Hasil evaluasi menunjukkan:
- LightGBM menjadi model dengan performa terbaik secara keseluruhan, unggul dalam semua metrik utama dan memberikan keseimbangan antara presisi dan recall.
- XGBoost memberikan kinerja yang sangat kompetitif, dengan tingkat overfitting yang rendah dan performa stabil di kelas minoritas.
- Random Forest juga tampil baik namun menunjukkan kecenderungan overfitting karena akurasi latih sangat tinggi.
- SVM dan Logistic Regression stabil, namun tidak efektif dalam mengenali kelas minoritas (CL).
- Decision Tree merupakan model dengan performa terendah dan paling rentan terhadap overfitting.

### 4. Tantangan pada Kelas Minoritas
Semua model masih memiliki kelemahan signifikan dalam mendeteksi kelas CL, yang jumlahnya sangat sedikit. Nilai recall dan f1-score untuk kelas ini umumnya sangat rendah. Hal ini menandakan perlunya pendekatan tambahan seperti:
- Oversampling (misal: SMOTE)
- Class weighting
- Pengumpulan data tambahan untuk kelas CL

### Secara keseluruhan, proyek ini berhasil mengembangkan model prediktif yang cukup kuat dan dapat digunakan sebagai dasar untuk pengembangan lebih lanjut. LightGBM dan XGBoost menjadi kandidat utama untuk implementasi model akhir. Ke depan, fokus bisa diarahkan pada penyeimbangan data antar kelas dan penyesuaian hyperparameter untuk lebih meningkatkan akurasi serta kinerja model terhadap kelas yang kurang terwakili.

### Rekomendasi
Berdasarkan seluruh hasil evaluasi dan perbandingan model:
XGBoost direkomendasikan untuk diimplementasikan dalam sistem prediksi sirosis hati karena:
- Memiliki performa paling tinggi dan stabil pada data latih maupun uji.
- Overfitting relatif terkendali.
- Menunjukkan kemampuan generalisasi yang baik terhadap data baru.
LightGBM juga merupakan alternatif kuat, namun perlu perhatian terhadap potensi overfitting dan peningkatan akurasi pada kelas minoritas.
