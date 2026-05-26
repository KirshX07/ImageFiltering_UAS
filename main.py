import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# =========================================================================
# 1. KONFIGURASI DATASET (Minimal 5 Gambar Daun dari Kamera HP)
# =========================================================================
# Pastikan kamu memiliki 5 foto ini di dalam folder yang sama dengan skrip kode ini
image_paths = ['foto1.jpg', 'foto2.jpg', 'foto3.jpg', 'foto4.jpg', 'foto5.jpg']

# Membuat direktori penyimpanan khusus untuk hasil eksplorasi laporan
os.makedirs('hasil_proses', exist_ok=True)

# Array log data kuantitatif untuk mencatat parameter statistik tiap metode
tabel_data = []

print("="*80)
print("SISTEM ANALISIS CITRA DIGITAL: DETEKSI TEPI & SEGMENTASI DAUN")
print("="*80)
print(f"Memulai pemrosesan secara berurutan terhadap {len(image_paths)} sampel citra...\n")

for idx, path in enumerate(image_paths):
    if not os.path.exists(path):
        print(f"[PERINGATAN] File '{path}' tidak ditemukan. Melewati file ini.")
        continue
        
    print(f"[PROSES] Mengeksekusi citra ke-{idx+1}: {path} ...")
    
    # Membaca citra asli dalam format BGR dan konversi ke RGB untuk Matplotlib
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # =========================================================================
    # 2. TAHAPAN PREPROCESSING CITRA (Sesuai Ketentuan Tugas)
    # =========================================================================
    # a. Resize Citra menjadi ukuran seragam kelas (512 x 512 piksel)
    img_resized = cv2.resize(img_rgb, (512, 512))
    total_piksel = img_resized.shape[0] * img_resized.shape[1] # 512 x 512 = 262.144 piksel
    
    # b. Grayscale Conversion (Mengubah citra warna menjadi skala keabuan 1-channel)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # c. Noise Removal Jalur 1: Gaussian Blur (Filter Spasial Linear Linear)
    img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # d. Noise Removal Jalur 2: Median Blur (Filter Non-Linear untuk Salt & Pepper)
    img_median = cv2.medianBlur(img_gray, 5)
    
    # Menggunakan representasi float32 dari Gaussian Blur untuk kalkulasi gradien orde-1
    gray_f = img_gaussian.astype(np.float32)
    
    # =========================================================================
    # 3. TAHAPAN EDGE DETECTION IMPLEMENTATION (4 Operator Komparatif)
    # =========================================================================
    # a. Operator Sobel (Kernel Spasial Pembobotan Pusat 3x3)
    sobel_x = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    edge_sobel = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
    
    # b. Operator Canny (Multi-stage Optimal Edge dengan Non-Maximum Suppression)
    edge_canny = cv2.Canny(img_gaussian, 50, 150)
    
    # c. Operator Prewitt (Kernel Spasial Konstan 3x3 Manual)
    kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(gray_f, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(gray_f, -1, kernel_prewitt_y)
    edge_prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.uint8)
    
    # d. Operator Roberts (Kernel Diagonal Silang Sempit 2x2 Manual)
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_x = cv2.filter2D(gray_f, -1, kernel_roberts_x)
    roberts_y = cv2.filter2D(gray_f, -1, kernel_roberts_y)
    edge_roberts = np.sqrt(roberts_x**2 + roberts_y**2).astype(np.uint8)
    
    # Ekstraksi Kuantitatif: Menghitung piksel bernilai tinggi di atas threshold gradien 35
    p_sobel = np.count_nonzero(edge_sobel > 35)
    p_canny = np.count_nonzero(edge_canny)
    p_prewitt = np.count_nonzero(edge_prewitt > 35)
    p_roberts = np.count_nonzero(edge_roberts > 35)
    
    # Memasukkan kalkulasi statistik ke dalam array log data untuk visualisasi tabel laporan
    tabel_data.append({
        'filename': path,
        'sobel': [p_sobel, (p_sobel / total_piksel) * 100, np.mean(edge_sobel)],
        'canny': [p_canny, (p_canny / total_piksel) * 100, np.mean(edge_canny)],
        'prewitt': [p_prewitt, (p_prewitt / total_piksel) * 100, np.mean(edge_prewitt)],
        'roberts': [p_roberts, (p_roberts / total_piksel) * 100, np.mean(edge_roberts)]
    })

    # =========================================================================
    # 4. TAHAPAN SEGMENTASI CITRA DAUN (Thresholding & K-Means)
    # =========================================================================
    # a. Otsu Thresholding (Pencarian Ambang Batas Biner Global Otomatis)
    _, seg_thresh = cv2.threshold(img_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # b. K-Means Clustering (Pengklasteran Ruang Warna RGB Asli dengan Nilai K=3)
    pixel_values = img_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[labels.flatten()]
    seg_kmeans = segmented_data.reshape(img_resized.shape)
    
    # =========================================================================
    # 5. GENERATE MULTI-PANEL GRID FILTER (Struktur Komparatif 3 x 4)
    # =========================================================================
    plt.figure(figsize=(16, 12))
    
    # BARIS 1: Preprocessing Pipeline Lengkap
    plt.subplot(3, 4, 1); plt.imshow(img_resized); plt.title('Original'); plt.axis('off')
    plt.subplot(3, 4, 2); plt.imshow(img_gray, cmap='gray'); plt.title('Grayscale'); plt.axis('off')
    plt.subplot(3, 4, 3); plt.imshow(img_gaussian, cmap='gray'); plt.title('Gaussian Blur'); plt.axis('off')
    plt.subplot(3, 4, 4); plt.imshow(img_median, cmap='gray'); plt.title('Median Blur'); plt.axis('off')
    
    # BARIS 2: Perbandingan Empat Metode Deteksi Tepi
    plt.subplot(3, 4, 5); plt.imshow(edge_sobel, cmap='gray'); plt.title('Edge: Sobel'); plt.axis('off')
    plt.subplot(3, 4, 6); plt.imshow(edge_canny, cmap='gray'); plt.title('Edge: Canny'); plt.axis('off')
    plt.subplot(3, 4, 7); plt.imshow(edge_prewitt, cmap='gray'); plt.title('Edge: Prewitt'); plt.axis('off')
    plt.subplot(3, 4, 8); plt.imshow(edge_roberts, cmap='gray'); plt.title('Edge: Roberts'); plt.axis('off')
    
    # BARIS 3: Hasil Pemisahan Objek Segmentasi & Kotak Keterangan Analisis
    plt.subplot(3, 4, 9); plt.imshow(img_resized); plt.title('Original (ref)'); plt.axis('off')
    plt.subplot(3, 4, 10); plt.imshow(seg_thresh, cmap='gray'); plt.title('Seg: Otsu Thresholding'); plt.axis('off')
    plt.subplot(3, 4, 11); plt.imshow(seg_kmeans); plt.title('Seg: K-Means (K=3)'); plt.axis('off')
    
    # Slot Panel Ke-12: Kotak Teks Ringkasan Analisis Visual Spasial
    plt.subplot(3, 4, 12)
    plt.title('ANALISIS SINGKAT', fontsize=11, fontweight='bold')
    text_content = (
        "Sobel   : Tepi jelas, sedikit noise\n"
        "Canny   : Tepi tipis & bersih\n"
        "Prewitt : Mirip Sobel, lebih halus\n"
        "Roberts : Sensitif, banyak noise\n\n"
        "Otsu    : Cepat, 2 kelas saja\n"
        "K-Means : 3 cluster warna"
    )
    plt.text(0.05, 0.5, text_content, fontsize=10, family='monospace',
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFFE0', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    output_filename = f'hasil_proses/output_citra_{idx+1}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> [SUKSES] Visualisasi grid gambar ke-{idx+1} disimpan ke '{output_filename}'")

# =========================================================================
# 6. AUTOMATIC EXPORT: GENERATE FOTO TABEL LAPORAN (.PNG)
# =========================================================================
print("\n" + "="*80)
print("PROSES GENERATE GAMBAR TABEL UTAL UTK WORD / GOOGLE DOCS")
print("="*80)

# 1. MEMBUAT FOTO TABEL 1: ANALISIS KUALITATIF SEGMENTASI CITRA
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
ax.axis('tight')

columns_seg = ['Metode Segmentasi', 'Kelebihan', 'Kekurangan', 'Pengaruh Pencahayaan & Background']
data_seg = [
    ['Otsu Thresholding', 'Komputasi sangat cepat (instan),\notomatis mencari ambang biner global.', 'Detail internal daun (tulang/serat)\nhilang, hasil biner kaku.', 'Sangat sensitif bayangan lampu\nruangan sensor HP.'],
    ['K-Means (K=3)', 'Sangat adaptif memisahkan area objek\nberdasarkan kemiripan warna RGB asli.', 'Komputasi iteratif lebih memakan waktu,\nsensitif titik centroid awal.', 'Lebih tangguh mereduksi degradasi\npencahayaan lampu ruangan.']
]

table_seg = ax.table(cellText=data_seg, colLabels=columns_seg, loc='center', cellLoc='left')
table_seg.auto_set_font_size(False)
table_seg.set_fontsize(10) # Menggunakan fungsi set_fontsize yang benar bebas dari error
table_seg.scale(1.2, 2.8)

# Memberi warna biru gelap pada header tabel segmentasi
for j in range(len(columns_seg)):
    table_seg[0, j].set_facecolor('#2E4053')
    table_seg[0, j].get_text().set_color('white')
    table_seg[0, j].get_text().set_weight('bold')

plt.savefig('hasil_proses/tabel_1_analisis_segmentasi.png', dpi=200, bbox_inches='tight')
plt.close()
print("1. File Gambar Berhasil Dibuat -> hasil_proses/tabel_1_analisis_segmentasi.png")

# 2. MEMBUAT FOTO TABEL 2: REKAPITULASI DATA KUANTITATIF DETEKSI TEPI DAUN
rows_edge = []
for data in tabel_data:
    fname = data['filename']
    methods = ['Sobel', 'Canny', 'Prewitt', 'Roberts']
    keys = ['sobel', 'canny', 'prewitt', 'roberts']
    for m, k in zip(methods, keys):
        piksel = f"{data[k][0]:,}"
        rasio = f"{data[k][1]:.2f}%"
        intensitas = f"{data[k][2]:.2f}"
        rows_edge.append([fname, m, piksel, rasio, intensitas])

fig, ax = plt.subplots(figsize=(11, 8))
ax.axis('off')
ax.axis('tight')

columns_edge = ['Nama File Gambar', 'Metode Edge', 'Jumlah Piksel Tepi', 'Rasio Tepi (%)', 'Rata-rata Intensitas']
table_edge = ax.table(cellText=rows_edge, colLabels=columns_edge, loc='center', cellLoc='center')
table_edge.auto_set_font_size(False)
table_edge.set_fontsize(9) # Menggunakan fungsi set_fontsize yang benar bebas dari error
table_edge.scale(1.1, 1.5)

# Memberi warna biru langit pada header tabel deteksi tepi
for j in range(len(columns_edge)):
    table_edge[0, j].set_facecolor('#1A5276')
    table_edge[0, j].get_text().set_color('white')
    table_edge[0, j].get_text().set_weight('bold')

# Styling baris data selang-seling per file gambar biar rapi dibaca dosen
for i in range(1, len(rows_edge) + 1):
    file_idx = (i - 1) // 4 
    if file_idx % 2 == 1:
        for j in range(len(columns_edge)):
            table_edge[i, j].set_facecolor('#EBF5FB')

plt.savefig('hasil_proses/tabel_2_analisis_kuantitatif.png', dpi=200, bbox_inches='tight')
plt.close()
print("2. File Gambar Berhasil Dibuat -> hasil_proses/tabel_2_analisis_kuantitatif.png")

print("\n" + "="*80)
print("[PROSES SELESAI] Semua asset laporanmu sudah siap dikumpulkan cik!")
print("="*80)