# 🍃 Leaf Morphology Analysis: Advanced Edge Detection & Segmentation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Sistem Analisis Citra Digital otomatis berbasis Python untuk mengekstrak, membandingkan, dan mengevaluasi fitur morfologi struktur daun menggunakan **4 Operator Deteksi Tepi** dan **2 Metode Segmentasi Citra**. Projek ini dikembangkan sebagai Tugas Akhir Individu untuk mata kuliah **Pengolahan Citra Digital (PCD)**.

---

## 📑 Daftar Isi
1. [Deskripsi Projek](#-deskripsi-projek)
2. [Arsitektur Pipeline Sistem](#-arsitektur-pipeline-sistem)
3. [Fitur Utama](#-fitur-utama)
4. [Struktur Direktori](#-struktur-direktori)
5. [Instalasi & Cara Menjalankan](#-instalasi--cara-menjalankan)
6. [Spesifikasi Output Visual (Grid 3x4)](#-spesifikasi-output-visual-grid-3x4)
7. [Ringkasan Hasil Analisis](#-ringkasan-hasil-analisis)
8. [Identitas Pengembang](#-identitas-pengembang)

---

## 📌 Deskripsi Projek
Projek ini melakukan pemrosesan *image processing pipeline* secara sekuensial terhadap 5 sampel foto daun (`foto1.jpg` hingga `foto5.jpg`) yang diambil secara mandiri menggunakan kamera HP. 

Sistem ini dirancang untuk memberikan analisis komparatif yang mendalam baik secara **Kualitatif** (melalui visualisasi plot grid multi-panel) maupun **Kuantitatif** (melalui kalkulasi otomatis jumlah piksel tepi, rasio persentase luasan, dan rata-rata intensitas gradien) yang diekspor langsung ke dalam bentuk grafik tabel siap pakai untuk laporan ilmiah.

---

## ⚙️ Arsitektur Pipeline Sistem

Sistem memproses setiap citra melalui alur komputasi spasial terstruktur sebagai berikut:

```text
  [ Citra Daun Asli (BGR) ]
              │
              ▼
     [ Konversi ke RGB ]
              │
              ▼
   [ Resize ke 512 x 512 px ] ───► Menyamakan beban komputasi & ukuran matriks kernel
              │
              ▼
  [ Grayscale Conversion ]    ───► Reduksi dari 3-channel ke 1-channel intensitas
              │
     ┌────────┴────────────────────────┐
     ▼                                 ▼
[ Gaussian Blur (5x5) ]          [ Median Blur (5x5) ]
(Reduksi Noise Frekuensi Tinggi)  (Eliminasi Salt-and-Pepper Noise)
     │
     ├─► [ DETEKSI TEPI ] ───────► Sobel (3x3), Prewitt (3x3), Roberts (2x2), Canny
     │
     └─► [ SEGMENTASI CITRA ] ───► Otsu Thresholding & K-Means Clustering (K=3)
```

---

## 🚀 Fitur Utama

* **Dual-Path Smoothing Preprocessing**: Mengombinasikan keunggulan *Gaussian Filter* dan *Median Filter* secara paralel untuk memastikan citra bersih dari derau latar belakang sebelum ekstraksi kontur dilakukan.
* **Comprehensive Edge Evaluation**: Implementasi komparatif lengkap dari operator gradien orde pertama berbasis bobot pusat (**Sobel**), bobot konstan (**Prewitt**), diferensial diagonal sempit (**Roberts**), hingga optimasi penekanan non-maksimum (**Canny**).
* **Multi-Class Color Segmentation**: Pemisahan objek menggunakan pendekatan ambang batas biner global (**Otsu Thresholding**) dibandingkan dengan pengklasteran ruang warna RGB adaptif berbasis optimasi jarak euclidian (**K-Means Clustering, K=3**).
* **Automated Report Asset Generation**: Sistem otomatis mendeteksi sebaran piksel, menghitung parameter statistik kuantitatif secara *real-time*, membenahi bug font size Matplotlib, dan mengekspor hasilnya menjadi file gambar tabel (`.png`) beresolusi tinggi (200 DPI).

---

## 📂 Struktur Direktori

```text
ImageFiltering_UAS/
│
├── main.py               # Skrip utama Python (Source Code)
├── foto1.jpg             # Sampel citra daun masukan 1
├── foto2.jpg             # Sampel citra daun masukan 2
├── foto3.jpg             # Sampel citra daun masukan 3
├── foto4.jpg             # Sampel citra daun masukan 4
├── foto5.jpg             # Sampel citra daun masukan 5
├── README.md             # Dokumentasi teknis repositori (File ini)
└── hasil_proses/         # Direktori luaran otomatis (Auto-generated)
    ├── output_citra_1.png   # Plot Grid Komparatif (3x4) untuk foto1
    ├── output_citra_2.png   # Plot Grid Komparatif (3x4) untuk foto2
    ├── output_citra_3.png   # Plot Grid Komparatif (3x4) untuk foto3
    ├── output_citra_4.png   # Plot Grid Komparatif (3x4) untuk foto4
    ├── output_citra_5.png   # Plot Grid Komparatif (3x4) untuk foto5
    ├── tabel_1_analisis_segmentasi.png  # Ekspor visual tabel perbandingan segmentasi
    └── tabel_2_analisis_kuantitatif.png # Ekspor visual data matriks piksel tepi
```

---

## 💻 Instalasi & Cara Menjalankan

### 1. Prasyarat Pustaka
Pastikan Anda telah memasang dependensi berikut di dalam lingkungan Python Anda:
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### 2. Eksekusi Program
Arahkan terminal ke folder direktori projek Anda, lalu jalankan perintah berikut:
```bash
cd C:/Users/kiran/OneDrive/Coding/ImageFiltering_UAS
python main.py
```
Sistem akan memproses ke-5 gambar daun secara berurutan dan menampilkan log progres pada terminal hingga selesai.

---

## 📊 Spesifikasi Output Visual (Grid 3x4)

Setiap gambar masukan menghasilkan sebuah file visualisasi terpadu berukuran **16 x 12 inci** dengan susunan panel matriks **3 Baris × 4 Kolom**:
* **Baris 1 (Preprocessing)**: `Original` ──► `Grayscale` ──► `Gaussian Blur` ──► `Median Blur`
* **Baris 2 (Deteksi Tepi)**: `Edge: Sobel` ──► `Edge: Canny` ──► `Edge: Prewitt` ──► `Edge: Roberts`
* **Baris 3 (Segmentasi & Log)**: `Original (ref)` ──► `Seg: Otsu Thresholding` ──► `Seg: K-Means (K=3)` ──► `Kotak Teks Analisis Singkat`

---

## 📈 Ringkasan Hasil Analisis

* **Deteksi Tepi Terbaik (Canny)**: Terbukti paling unggul untuk pemetaan morfologi kontur daun. Algoritma Canny menghasilkan garis tepi yang paling ramping (*thinning* tepat 1 piksel) serta memiliki imunitas tertinggi terhadap tekstur *noise* pada permukaan meja kerja berkat fungsi ambang batas ganda (*Hysteresis*).
* **Segmentasi Citra Terbaik (K-Means K=3)**: Sangat tangguh dan adaptif dalam mengisolasi bodi daun dari latar belakangnya. Dengan mengelompokkan warna pada ruang spektral RGB asli, K-Means sukses memisahkan objek daun dari bayangan lampu ruangan maupun degradasi pencahayaan yang tidak merata, mengungguli metode Otsu biner global yang cenderung kaku.

---

## 👤 Identitas Pengembang

* **Nama Pengembang** : Kirana Shofa Dzakiyyah
* **NIM** : 25051204358
* **Program Studi** : S1 Teknik Informatika
* **Fakultas** : Fakultas Teknik
* **Instansi** : Universitas Negeri Surabaya (UNESA)
* **Dosen Pengampu** : Martini Dwi Endah Susanti, S.Kom., M.Kom.
* **Repositori GitHub** : [KirshX07/ImageFiltering_UAS](https://github.com/KirshX07/ImageFiltering_UAS)
