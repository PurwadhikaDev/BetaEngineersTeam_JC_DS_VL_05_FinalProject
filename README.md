# MACHINE LEARNING MODEL FOR PREDICTING PROPERTY VALUE IN PHILADELPHIA
Created By : BETA ENGINEERS (PURWADHIKA)

Team Members :
1.  Yehezkiel Gabriel Sutopo
2.  Yohanna Inawati Santoso
3.  Risdan Kristori

Data Source : [Philadelphia - Buildings Database](https://www.kaggle.com/datasets/adebayo/philadelphia-buildings-database?select=PHL_OPA_PROPERTIES.csv)

Python Library Versions :
- numpy 1.22.2
- pandas 1.4.1
- matplotlib 3.5.1
- seaborn 0.11.2
- sickit-learn 1.1.1
- category_encoders 2.5.0
- xgboost 1.6.1

<img src="pic/philadelphia.jpg" alt="isolated" width="1080"/>

# Contents

1.  Business Problem Understanding
2.  Data Understanding
3.  Data Preprocessing
4.  Modeling
5.  Conclusion
6.  Recommendation

## Business Problem Understanding

**Context**

Office of Property Assessment (OPA) merupakan salah satu departemen pada pemerintahan di Kota Philadelphia yang bertugas untuk menentukan nilai dari seluruh properti yang ada di Kota Philadelphia. Nilai dari setiap properti di Kota Philadelphia tersebut bermanfaat sebagai dasar pertimbangan masyarakat dalam menentukan harga transaksi properti milik mereka. Selain itu, nilai properti ini juga bermanfaat untuk menentukan nilai pajak dari setiap properti yang harus dibayarkan oleh pemilik properti tersebut, dimana pajak dari properti merupakan penyumbang terbesar dalam pendanaan sekolah umum di Kota Philadelphia.

Reference:  [https://www.phila.gov/departments/office-of-property-assessment/](https://www.phila.gov/departments/office-of-property-assessment/)

**Problem Statement**
-   Tipe properti apa saja yang memerlukan perhatian lebih dalam pelaksanaan quality control?
-   Tipe zoning properti apa yang memiliki jumlah terbanyak dan memiliki nilai properti tertinggi di Kota Philadelphia?
-   Variabel apakah yang paling berpengaruh terhadap nilai properti di Kota Philadelphia?
-   Model machine learning apakah yang dapat membantu OPA dalam memprediksi nilai properti dengan baik?

**Goals**

Berdasarkan permasalahan tersebut, OPA memerlukan analisis mendalam terkait nilai properti yang ada di Kota Philadelphia dan sebuah 'tool' yang dapat membantu mereka dalam menentukan nilai properti yang tepat berdasarkan karakteristik dari masing-masing properti.

**Analytic Approach**

Jadi, yang akan dilakukan adalah menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada dalam menentukan nilai properti berdasarkan karakteristiknya. Selanjutnya, kita akan membangun suatu model regresi yang akan membantu OPA untuk menentukan nilai properti tersebut.

**Metric Evaluation**

Dalam memilih model terbaik yang akan digunakan untuk memprediksi nilai properti perlu adanya matrix evaluasi yang sesuai. Pada data ini kami menggunakan matrix Mean Absolut Percentrage Error (MAPE) dan R-Squared. 
Penggunaan MAPE dikarenakan banyaknya nilai outlier yang terkandung di dalam data dan rentang nilai properti yang tinggi (1300 - 35214380), hal ini wajar karena di dalam data terkandung berbagai macam jenis properti mulai dari jenis residential hingga spesial properti seperti stadiun.

MAPE merupakan besaran yang mengukur residual (error) antara nilai hasil prediksi dengan nilai sebenarnya dalam besaran persen. Nilai residual yang dihitung adalah nilai absolut dan dibagi dengan nilai sebenarnya dikalikan 100 persen.
Berikut adalah kategori nilai MAPE:

1.  0-10% : Sangat baik
2.  10-20% : Baik
3.  20-50% : Wajar
4.  '>50% : Tidak akurat

R-squared merupakan suatu nilai yang memperlihatkan seberapa besar variabel independen mempengaruhi variabel dependen. R-squared merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen. Nilai R-squared (R2) digunakan untuk menilai seberapa besar pengaruh variabel independen tertentu terhadap variabel dependen.
Terdapat tiga kategori pengelompokan pada nilai R-squared yaitu kategori kuat, kategori moderat, dan kategori lemah. Hair et al menyatakan bahwa nilai R squared 0,75 termasuk ke dalam kategori kuat, nilai R-squared 0,50 termasuk kategori moderat dan nilai R squared 0,25 termasuk kategori lemah.

## Data Understanding

-   Dataset merupakan data karakteristik, harga sales, dan nilai properti yang ada di Kota Philadelphia hingga tahun 2020.
-   Setiap baris data merepresentasikan informasi terkait masing-masing properti.
- Terdapat 75 fitur pada dataset PHL OPA Properties. 

## Data Preprocessing

Sebelum masuk ke dalam tahap modeling, maka akan dilakukan pre-processing. Berikut adalah tahapan pre-processing yang dilakukan pada model ini:

**1. Menghapus fitur yang tidak berguna untuk proses berikutnya**
- Fitur tergolong unique
- Fitur dengan null value >50% yang tidak dapat diimpute, 
- Fitur dengan makna sama atau berisi deskripsi dari fitur sebelumnya
- Fitur yang diperoleh dari hasil market value
 
**2. Imputer (Menangani Missing value)**

-   Menginput nilai nan pada kolom  **fireplaces**,  **garage_spaces**,  **interior_condition**, dan  **exterior_condition**  dengan value 0
-   Menginput nilai nan pada kolom  **basements**,  **garage_type**,  **type_heater**  **view_type**, dan  **separate_utilities**  dengan value 'O'
-   Menginput nilai nan pada kolom  **topography**  dan  **unfinished**  dengan value 'F'
-   Menginput nilai nan pada kolom  **other_building**  dengan value 'N'
-   Menginput nilai nan pada kolom  **street_designation**  dengan value 'WHRF'

**3.  Encoding**

-   Mengubah fitur  **garage_type**,  **type_heater**,  **view_type**,  **zoning**,  **street_designation**, dan  **topography**  menggunakan Binary Encoding, karena fitur-fitur ini memiliki unique value yang banyak sehingga akan menghasilkan jumlah kolom yang terlalu banyak apabila menggunakan OneHotEncoding.
-   Merubah fitur  **unfinished**,  **other_building**,  **basements**,  **separate_utilities**,  **category_code**  dengan menggunakan OneHotEncoding, karena fitur memiliki unique value yang sedikit dan tidak memiliki urutan/ordinal.

**4.  Scaler**

-   Melakukan scaling pada fitur  **depth**,  **frontage**,  **house_extension**,  **number_of_bathrooms**,  **number_of_bedrooms**,  **number_of_rooms**,  **number_stories**,  **total_area**,  **total_livable_area**,  **sale_price**  dengan menggunakan RobustScaler, karena fitur-fitur di atas memiliki interval yang lebar dan outlier didalamnya.

## Modeling

Membandingkan Linear Regression, Random Forest Regressor dan XGBoost Regressor.
Berikut ini list perbandingan yang telah dilakukan :
| **Model Description** | **Data Set** | **MAPE Score** | **RSquared Score** |
| --- | --- | --- | --- |
| Linear Regression | Train | -1.182605e+14 | 2.365211e+14 |
| Random Forest Regressor | Train | -2.042737e-01 | 2.810548e-03 |
| XGBoost Regressor | Train | -2.401568e-01 | 2.057117e-03 |
| Random Forest Regressor | Test | 0.238641 | 0.780251 |
| XGBoost Regressor | Test | 0.361665 | 0.782011 |
| XGBoost + Hyperparameter | Train | -0.243690 | - |
| XGBoost + Hyperparameter | Test | 0.231748 | 0.783908 |
| XGBoost + Feature Importance | Test | 0.239831 | 0.788639 |

Hasilnya adalah XGBoost dengan hyperparameter tuning merupakan model terbaik dengan score MAPE 0.231748 dan Rsquare 0.783908.

## Conclusion

1.  Properti berkategori commercial, multi family, dan industrial merupakan kategori properti yang memiliki nilai outlier terbanyak pada market_valuenya.
2.  Tipe properti yang memiliki jumlah terbanyak di Kota Philadelphia adalah tipe zoning RSA (Residential Single Family) dan RMA (Residential Multifamily). Sedangkan, Tipe properti yang memiliki nilai median market_value tertinggi adalah tipe zoning special purpose.
3.  Berdasarkan nilai korelasi spearman, 3 feature numerik yang memiliki korelasi tertinggi terhadap market_value adalah total_livable_area, sale_price, dan frontage. Sementara, berdasarkan model yang dibangun, 3 feature yang memiliki pengaruh paling dominan terhadap market_value adalah category_code, total_livable_area, dan number_stories.
4.  Model yang telah dibangun memiliki score MAPE sebesar 23%. yang berarti ketika model yang dibuat digunakan untuk memprediksi nilai properti pada rentang nilai seperti yang dilatih terhadap model (market value: 1,300 USD - 352,143,800 USD), maka hasil prediksi yang dihasilkan oleh model memiliki kemungkinan tingkat kesalahan sebesar 23% dari nilai aslinya. Dengan nilai MAPE sebesar 23% menjadikan model ini sebagai model yang menghasilkan nilai prediksi yang wajar.
5.  Model yang telah dibangun memiliki score R-squared sebesar 78,3% yang berarti model yang telah dibangun mampu menjelaskan faktor-faktor yang mempengaruhi market_value sebesar 78,3%.


## Recommendation

Bagi pemerintah Kota Philadelphia:
1.  Dengan nilai market_value yang telah ditentukan, maka pemerintah Kota Philadelphia dapat memprediksi pendapatan yang dihasilkan dari pajak properti pada Kota Philadelphia. Hal tersebut dapat menjadi acuan bagi pemerintah untuk membuat anggaran Kota Philadelphia khususnya pada bidang pendidikan untuk periode selanjutnya.
2.  Dengan menetapkan nilai pajak properti berdasarkan nilai market_value sebuah properti yang telah sesuai dengan keadaan setiap properti. Maka, pemerintah dapat memperkuat kebijakan terkait pembayaran pajak serta mendorong masyarakat untuk taat membayar pajak. Hal tersebut diharapkan dapat meningkatkan persentase tingkat pembayaran pajak tahunan.

Bagi OPA:
1.  Berdasarkan hasil dari model yang telah dibangun. OPA dapat menjelaskan kepada masyarakat Kota Philadelphia mengenai faktor-faktor yang mempengaruhi nilai dari masing-masing properti sehingga masyarakat dapat memahami apakah properti miliknya memiliki nilai yang lebih rendah/tinggi dari yang seharusnya.
2.  Dalam melakukan quality control, OPA dapat memberikan perhatian lebih terhadap properti yang berkategori commercial, multi family, dan industrial karena kategori tersebut cenderung memiliki outlier yang tinggi.

Bagi Investor/pelaku bisnis:
1.  Nilai market value dapat digunakan sebagai acuan karakteristik penduduk pada area tersebut, sehingga Investor/pelaku bisnis dapat menentukan lokasi bisnis yang sesuai dengan target market mereka.

Untuk Model selanjutnya:
1.  Hasil benchmark model RandomForest menunjukkan hasil yang lebih baik dibandingkan dengan XGBoost. Penerapan Hyperparameter pada model RandomForest mungkin saja dapat menghasilkan model dengan score yang lebih baik.
2.  Menambahkan feature-feature baru yang dapat menjelaskan market_value khususnya pada properti dengan kategori industrial, commercial, dan multifamily.
3.  Score yang lebih baik mungkin dapat dicapai dengan menggunakan algoritma machine learning yang lain. Sehingga, perlu adanya prediksi menggunakan model regresi selain Linear Regression, Random Forest Regressor, dan XG Boost Regressor.

