"""
Dashboard Streamlit - Klasifikasi Citra Organisme Laut (OceanicLife) dengan CNN
Jalankan:  streamlit run app.py
Pastikan folder `artifacts/` (hasil notebook) berada di folder yang sama dengan file ini.
"""
import os
os.environ.setdefault("KERAS_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Catatan visual karakteristik tiap kelas organisme laut
CATATAN_VISUAL = {
    "coral": "Struktur kalsium karbonat bercabang/berbatu warna-warni pada substrat dasar laut.",
    "crab": "Krustasea berkaki sepuluh dengan sepasang capit kokoh dan karapas keras.",
    "dolphin": "Mamalia laut ramping dengan moncong memanjang, sirip punggung melengkung, dan sering terlihat di permukaan air.",
    "eel": "Ikan berbadan panjang seperti ular tanpa sirip perut, sering bersembunyi di celah karang.",
    "fish": "Ikan bertulang sejati dengan sisik, sirip dada, dan sirip ekor aktif di kolom air.",
    "jellyfish": "Tubuh transparan berbentuk payung (lonceng) dengan tentakel menjuntai.",
    "octopus": "Moluska tanpa cangkang bertubuh lunak dengan delapan lengan berpenghisap kuat.",
    "ray": "Ikan bertubuh pipih melebar seperti sayap dengan ekor panjang, sering melayang dekat dasar laut.",
    "seahorse": "Ikan kecil berkepala menyerupai kuda, berpostur tegak dengan ekor melingkar.",
    "seal": "Mamalia laut semi-akuatik berkaki sirip dan lapisan lemak tebal di pesisir bebatuan.",
    "shark": "Predator bertubuh torpedo dengan sirip punggung segitiga kaku dan celah insang lateral.",
    "shrimp": "Krustasea kecil bertubuh melengkung dengan antena panjang dan kaki renang.",
    "starfish": "Echinodermata berbodi pipih dengan simetri radial 5 lengan berduri di permukaan pasir/batu.",
    "turtle": "Reptil laut bercangkang streamline dengan tungkai depan berbentuk dayung renang.",
    "other": "Gabungan organisme laut lain di luar 8 kelas utama (mis. lumba-lumba, paus, ubur-ubur, pari, belut, kuda laut, udang), sehingga visualnya sangat beragam.",
    "whale": "Mamalia laut berukuran sangat besar dengan ekor mendatar (fluke) dan lubang semprot di punggung.",
}

st.set_page_config(page_title="Klasifikasi Organisme Laut - CNN", page_icon="🐠", layout="wide")


# ------------------------------------------------------------------ loader
@st.cache_resource
def load_model():
    return keras.models.load_model(ARTIFACT_DIR / "model.keras")


@st.cache_data
def load_eval():
    with open(ARTIFACT_DIR / "eval_data.pkl", "rb") as f:
        return pickle.load(f)


if not (ARTIFACT_DIR / "model.keras").exists() or not (ARTIFACT_DIR / "eval_data.pkl").exists():
    st.error("Folder `artifacts/` belum lengkap. Jalankan dulu notebook `CNN_OceanicLife_Klasifikasi.ipynb` "
             "sampai selesai, lalu letakkan `app.py` di folder yang sama dengan folder `artifacts/`.")
    st.stop()

model = load_model()
E = load_eval()
class_names = E["class_names"]
IMG_SIZE = E["img_size"]
hist = E["history"]
cm = E["cm"]
per_class = E["per_class_df"]
metrics = E["metrics_df"].set_index("Metrik Evaluasi")["Nilai"]
split = E["split_info"]
counts = E["class_counts"]

if len(class_names) < 2:
    st.error("Artefak hanya memuat kurang dari 2 kelas, sehingga hasilnya tidak valid. "
             "Periksa label dataset lalu jalankan ulang notebook.")
    st.stop()
if E.get("label_source") == "zeroshot":
    st.warning("Pada praktikum ini label datanya berasal dari **CLIP zero-shot (pseudo-label)**, bukan label asli karena data dari kaggle nya tidak memiliki label. Jadi metrik di bawah ini mengukur "
               "kesesuaian CNN dengan label CLIP, bukan dari akurasi terhadap kebenaran sebenarnya.")
for _w in E.get("sanity_warnings", []):
    if "zero-shot" not in _w:
        st.warning(_w)


# ------------------------------------------------------------------ helper
def tingkat(v):
    return "sangat baik" if v >= 0.9 else "baik" if v >= 0.8 else "cukup" if v >= 0.7 else "masih perlu ditingkatkan"


def show_png(name, caption=None):
    p = ARTIFACT_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Gambar `{name}` tidak ditemukan di folder artifacts.")


def preprocess(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    return tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0   # identik dengan pipeline pelatihan


def make_gradcam(img_array, class_index):
    names = [l.name for l in model.layers]
    last_conv = "out_relu" if "out_relu" in names else [l for l in model.layers
                                                         if isinstance(l, keras.layers.Conv2D)][-1].name
    grad_model = keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    x = tf.convert_to_tensor(img_array[np.newaxis, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def overlay(img_array, heat, alpha=0.45):
    hm = Image.fromarray(np.uint8(255 * heat)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    hm = plt.cm.jet(np.array(hm) / 255.0)[..., :3]
    return np.clip((1 - alpha) * img_array + alpha * hm, 0, 1)


# ------------------------------------------------------------------ analisis otomatis
def analisis_distribusi():
    ratio = E["imbalance_ratio"]
    status = ("tidak seimbang (imbalance tinggi)" if ratio > 3
              else "cukup tidak seimbang (imbalance ringan-sedang)" if ratio > 1.5 else "relatif seimbang")
    teks = (f"Kelas dengan gambar terbanyak adalah **{counts.idxmax()}** ({counts.max()} gambar), sedangkan yang "
            f"tersedikit adalah **{counts.idxmin()}** ({counts.min()} gambar). Rasio keduanya {ratio:.2f} : 1, "
            f"sehingga distribusi data {status}.")
    if E["class_weight_used"]:
        teks += (" Karena ada ketidakseimbangan, pelatihan memakai *class weight* agar kelas minoritas tidak "
                 "diabaikan model.")
    else:
        teks += " Class weight tidak diperlukan karena proporsi antar kelas sudah cukup merata."
    return teks


def analisis_ukuran():
    s = E["img_stats"]
    modes = ", ".join(f"{k}: {v}" for k, v in s["modes"].items())
    teks = (f"Dari {s['n_sampled']} gambar sampel, lebar berkisar {s['width_min']}-{s['width_max']} px "
            f"(median {s['width_median']:.0f}) dan tinggi {s['height_min']}-{s['height_max']} px "
            f"(median {s['height_median']:.0f}). Mode warna: {modes}. ")
    if s["width_min"] != s["width_max"] or s["height_min"] != s["height_max"]:
        teks += f"Ukuran gambar tidak seragam, sehingga semua gambar di-*resize* menjadi {IMG_SIZE}x{IMG_SIZE} piksel. "
    else:
        teks += "Ukuran gambar sudah seragam. "
    if set(s["modes"]) - {"RGB"}:
        teks += "Gambar non-RGB (grayscale/palet/RGBA) dikonversi ke 3 kanal RGB saat pra-pemrosesan. "
    teks += (f"Gambar rusak yang dibuang sebelum pelatihan: {s['n_corrupt']}; duplikat dibuang: {s.get('n_duplicate', 0)}; "
             f"dilewati karena format tidak didukung: {s.get('n_skipped_format', 0)}.")
    return teks


def analisis_training():
    val_acc_all = np.array(hist["val_accuracy"])
    b = int(np.argmax(val_acc_all))
    tr_acc, va_acc = hist["accuracy"][b], hist["val_accuracy"][b]
    gap = tr_acc - va_acc
    n_ep = len(val_acc_all)
    fs = E.get("finetune_start", 0)
    teks = (f"Pelatihan berjalan {n_ep} epoch; akurasi validasi tertinggi **{va_acc:.2%}** dicapai pada **epoch {b + 1}** "
            f"(akurasi latih pada epoch itu {tr_acc:.2%}). ")
    if 0 < fs < n_ep:
        a1, a2 = max(val_acc_all[:fs]), max(val_acc_all[fs:])
        teks += (f"Garis putus-putus menandai awal *fine-tuning*. Akurasi validasi terbaik fase 1 (hanya melatih lapisan "
                 f"klasifikasi) adalah {a1:.2%}, sedangkan fase 2 (fine-tuning) {a2:.2%}"
                 + (", sehingga fine-tuning memberi peningkatan. " if a2 > a1 else ", sehingga fine-tuning tidak memberi peningkatan berarti. "))
    if va_acc < 0.6 and tr_acc < 0.6:
        teks += "Kedua akurasi masih rendah, indikasi *underfitting*: tambah epoch atau buka lebih banyak layer saat fine-tuning."
    elif gap > 0.10:
        teks += "Selisih akurasi latih dan uji cukup besar, indikasi *overfitting*: perkuat augmentasi/dropout atau tambah data."
    elif gap < -0.02:
        teks += ("Akurasi uji lebih tinggi daripada akurasi latih; ini wajar karena augmentasi dan dropout hanya aktif saat "
                 "pelatihan sehingga data latih terasa lebih sulit.")
    else:
        teks += "Selisih akurasi latih dan uji kecil, sehingga model *generalisasi dengan baik*."
    if hist["loss"][-1] < 1e-4:
        teks += " **Peringatan:** loss latih ≈ 0, tidak wajar; periksa label dan jumlah kelas."
    if hist["loss"][-1] < hist["loss"][0]:
        teks += " Loss menurun seiring epoch, menandakan model konvergen."
    return teks


def top_confusions(n=3):
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    return sorted(pairs, reverse=True)[:n]


def analisis_cm():
    recall = np.diag(cm) / cm.sum(axis=1)
    best, worst = int(np.argmax(recall)), int(np.argmin(recall))
    teks = (f"Diagonal utama menunjukkan prediksi yang benar. Kelas paling akurat dikenali: "
            f"**{class_names[best]}** (recall {recall[best]:.2%}); paling sulit: **{class_names[worst]}** "
            f"(recall {recall[worst]:.2%}). ")
    conf = top_confusions()
    if conf:
        teks += "Kesalahan terbanyak: " + "; ".join(
            f"{a} salah dikenali sebagai {b} ({n} gambar)" for n, a, b in conf) + ". "
        teks += "Kelas yang sering tertukar kemungkinan memiliki kemiripan bentuk, warna, atau latar belakang."
    else:
        teks += "Tidak ada kesalahan klasifikasi pada data uji."
    return teks


def analisis_metrik():
    acc, prec, rec, f1 = (metrics["Accuracy"], metrics["Precision (macro)"],
                          metrics["Recall (macro)"], metrics["F1-Score (macro)"])
    best, worst = per_class["f1-score"].idxmax(), per_class["f1-score"].idxmin()
    return (f"Akurasi {acc:.2%} (kategori: {tingkat(acc)}); semakin tinggi akurasi, semakin baik model mengenali kelas "
            f"organisme laut. F1-Score macro {f1:.2%} (kategori: {tingkat(f1)}) menunjukkan keseimbangan antara presisi "
            f"({prec:.2%}) dan recall ({rec:.2%}). Kelas terbaik: **{best}** (F1 {per_class.loc[best, 'f1-score']:.2%}); kelas tersulit: "
            f"**{worst}** (F1 {per_class.loc[worst, 'f1-score']:.2%}, {int(per_class.loc[worst, 'support'])} gambar uji).")


def analisis_prediksi(probs):
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else None
    conf = probs[top]
    teks = f"Model memprediksi **{class_names[top]}** dengan keyakinan **{conf:.1%}**. "
    if conf >= 0.8:
        teks += "Keyakinan tinggi: ciri visual gambar sangat sesuai dengan pola kelas ini. "
    elif conf >= 0.5:
        teks += "Keyakinan sedang: prediksi masuk akal namun ada kemungkinan kelas lain. "
    else:
        teks += "Keyakinan rendah: gambar mungkin ambigu, berkualitas kurang baik, atau berada di luar kelas yang dilatih. "
    if second is not None and probs[second] > 0.2:
        teks += f"Kandidat terdekat berikutnya adalah {class_names[second]} ({probs[second]:.1%}). "
    f1_cls = per_class.loc[class_names[top], "f1-score"]
    teks += f"Pada data uji, kelas ini memiliki F1-Score {f1_cls:.2%}, sehingga model tergolong {tingkat(f1_cls)} pada kelas ini."
    return teks


# ------------------------------------------------------------------ tampilan
st.title("🐠 Klasifikasi Citra Organisme Laut dengan CNN")
st.markdown(
    "Dashboard ini menampilkan hasil eksperimen *Convolutional Neural Network* (CNN) **MobileNetV2** (transfer learning) untuk mengklasifikasikan "
    f"gambar ke dalam **{len(class_names)} kelas organisme laut** pada dataset **OceanicLife**. Mulai dari "
    "ringkasan dataset, eksplorasi data, proses pelatihan, evaluasi model, hingga uji prediksi gambar baru secara langsung."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📦 Ringkasan Dataset", "📊 EDA", "🧠 Model & Pelatihan", "✅ Evaluasi", "🔍 Prediksi Gambar"])

# ---- Tab 1
with tab1:
    total = int(counts.sum())
    same_set = E.get("val_equals_test", split["n_val"] == split["n_test"])
    if same_set:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Kelas", len(class_names))
        c2.metric("Total Gambar", f"{total:,}")
        c3.metric("Data Latih (80%)", f"{split['n_train']:,}")
        c4.metric("Data Validasi/Uji (20%)", f"{split['n_test']:,}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jumlah Kelas", len(class_names))
        c2.metric("Total Gambar", f"{total:,}")
        c3.metric("Data Latih", f"{split['n_train']:,}")
        c4.metric("Data Validasi", f"{split['n_val']:,}")
        c5.metric("Data Uji", f"{split['n_test']:,}")
    st.caption(f"Sumber label: **{E.get('label_source', '-')}**")
    if E.get("label_source") == "zeroshot":
        hp_ = E.get("hyperparams", {})
        st.info("Dataset mentah tidak memiliki folder kelas, sehingga label dibuat otomatis oleh CLIP (zero-shot). "
                f"Ambang keyakinan CLIP: {hp_.get('Ambang Keyakinan CLIP', '-')}; "
                f"gambar dibuang karena keyakinan rendah: {hp_.get('Gambar Dibuang (Keyakinan CLIP Rendah)', '-')}; "
                f"gambar yang masuk kelas 'other' (organisme laut lain): {hp_.get('Gambar Masuk Kelas Other', '-')}.")
    st.markdown(f"**Daftar kelas:** {', '.join(class_names)}")

    st.subheader("Jumlah gambar per kelas pada data latih dan uji")
    sp = split["per_class"]
    if same_set:
        sp = sp[["Train", "Test"]].rename(columns={"Train": "Latih (80%)", "Test": "Validasi/Uji (20%)"})
    st.dataframe(sp.assign(Total=sp.sum(axis=1)), width="stretch")
    n_tr, n_te = split["n_train"], split["n_test"]
    if same_set:
        teks = (f"**Analisis:** Dataset berisi {total:,} gambar valid dan unik dari {len(class_names)} kelas, dibagi secara "
                f"*stratified* menjadi {n_tr:,} data latih ({n_tr / total:.0%}) dan {n_te:,} data validasi/uji "
                f"({n_te / total:.0%}) sesuai ketentuan praktikum 80:20. Seperti diminta praktikum, 20% data berperan sebagai "
                "satu *Validation/Test Set* yang sama: bobot model tidak pernah dilatih pada data ini, tetapi data ini dipakai "
                "untuk memantau pelatihan (EarlyStopping) dan evaluasi akhir, sehingga metrik dapat sedikit lebih optimistis "
                "daripada data yang benar-benar terpisah. Pembagian dilakukan sebelum augmentasi dan duplikat gambar sudah "
                "dibuang sehingga tidak ada gambar yang sama pada data latih dan uji.")
    else:
        teks = (f"**Analisis:** Dataset berisi {total:,} gambar valid dan unik yang dibagi secara *stratified* menjadi "
                f"{n_tr:,} latih ({n_tr / total:.0%}), {split['n_val']:,} validasi ({split['n_val'] / total:.0%}), dan "
                f"{n_te:,} uji ({n_te / total:.0%}). Pembagian dilakukan sebelum augmentasi dan duplikat sudah dibuang "
                "sehingga tidak terjadi kebocoran data.")
    st.markdown(teks)

# ---- Tab 2
with tab2:
    st.subheader("Distribusi data per kelas")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index, counts.values, color=plt.cm.viridis(np.linspace(0.15, 0.9, len(counts))))
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Jumlah gambar")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
    st.markdown("**Analisis:** " + analisis_distribusi())

    st.subheader("Contoh gambar dari setiap kelas")
    show_png("contoh_gambar.png")
    teks = ("**Analisis:** Setiap baris adalah satu kelas dan tiap kolom adalah gambar acak yang berbeda dari kelas tersebut. "
            "Variasi sudut pandang, pencahayaan, warna air, dan latar belakang yang terlihat antar-gambar dalam satu "
            "kelas menjadi tantangan bagi CNN, sehingga augmentasi dan normalisasi diperlukan.")
    st.markdown(teks)
    present = {c.lower() for c in class_names}
    for k, v in CATATAN_VISUAL.items():
        if k.lower() in present:
            st.markdown(f"- **{k.capitalize()}:** {v}")

    st.subheader("Ukuran dan format gambar")
    show_png("ukuran_gambar.png")
    st.markdown("**Analisis:** " + analisis_ukuran())

    st.subheader("Contoh hasil augmentasi data")
    show_png("augmentasi.png")
    st.markdown("**Analisis:** Augmentasi (flip horizontal, rotasi, zoom, dan pergeseran) membuat gambar latih bervariasi "
                "setiap epoch tanpa menambah file baru. Ini mengurangi overfitting karena model dipaksa mengenali objek "
                "dari berbagai posisi dan skala. Augmentasi hanya diterapkan pada data latih, tidak pada data uji.")

# ---- Tab 3
with tab3:
    st.subheader("Arsitektur dan parameter model")
    hp = pd.DataFrame(list(E["hyperparams"].items()), columns=["Parameter", "Nilai"]).astype(str)
    st.dataframe(hp, width="stretch", hide_index=True)
    st.markdown("**Analisis:** Model memakai **MobileNetV2**, arsitektur CNN yang dilatih sebelumnya pada dataset ImageNet "
                "(*transfer learning*). Lapisan convolution-nya sudah mampu mengenali tepi, tekstur, dan bentuk umum, "
                "sehingga hanya lapisan klasifikasi di atasnya (Global Average Pooling, Dropout, dan Dense Softmax) yang "
                "perlu dilatih pada fase 1. Pada fase 2 (*fine-tuning*), beberapa layer terakhir dibuka dan dilatih dengan "
                "learning rate sangat kecil agar fitur disesuaikan dengan citra organisme laut tanpa merusak bobot awal.")

    st.subheader("Grafik akurasi dan loss selama pelatihan")
    ep = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, hist["accuracy"], "o-", label="Latih")
    axes[0].plot(ep, hist["val_accuracy"], "s-", label="Validasi")
    axes[0].set(title="Akurasi", xlabel="Epoch", ylabel="Akurasi")
    axes[1].plot(ep, hist["loss"], "o-", label="Latih")
    axes[1].plot(ep, hist["val_loss"], "s-", label="Validasi")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    fs = E.get("finetune_start", 0)
    for ax in axes:
        if 0 < fs < len(hist["loss"]):
            ax.axvline(fs + 0.5, color="gray", linestyle="--", label="Mulai fine-tuning")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Analisis:** " + analisis_training())

# ---- Tab 4
with tab4:
    st.subheader("Hasil evaluasi model pada data validasi/uji (tidak dipakai untuk melatih bobot model)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    m2.metric("Precision (macro)", f"{metrics['Precision (macro)']:.2%}")
    m3.metric("Recall (macro)", f"{metrics['Recall (macro)']:.2%}")
    m4.metric("F1-Score (macro)", f"{metrics['F1-Score (macro)']:.2%}")
    st.dataframe(E["metrics_df"].round(4), width="stretch", hide_index=True)
    st.markdown("**Analisis:** " + analisis_metrik())
    base = E.get("baseline_acc")
    if base is not None:
        st.markdown(f"**Pembanding (baseline):** selalu menebak kelas mayoritas menghasilkan akurasi **{base:.2%}**. "
                    f"Model memperoleh {metrics['Accuracy']:.2%}, yaitu selisih {metrics['Accuracy'] - base:+.2%}.")
    if metrics["Accuracy"] >= 0.999:
        st.warning("Akurasi ≥ 99,9% sangat jarang pada data nyata. Periksa kebocoran data, duplikat mirip, atau label yang terlalu mudah.")

    st.subheader("Metrik per kelas")
    st.dataframe(per_class.round(4), width="stretch")
    fig, ax = plt.subplots(figsize=(10, 4))
    per_class[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0, 1.05); plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Analisis:** Perbandingan antar kelas memperlihatkan kelas mana yang stabil dan mana yang sulit. "
                "Recall rendah berarti banyak gambar kelas tersebut tidak terdeteksi; presisi rendah berarti model "
                "terlalu sering salah menuduh gambar lain sebagai kelas tersebut.")

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("**Analisis:** " + analisis_cm())

    st.subheader("Feature importance (Grad-CAM)")
    show_png("gradcam.png")
    st.markdown("**Analisis:** Heatmap Grad-CAM menandai area gambar yang paling memengaruhi keputusan CNN (merah = "
                "pengaruh tinggi, biru = rendah). Jika area merah berada pada tubuh organisme, model belajar dari fitur "
                "yang tepat; jika berada pada latar belakang, model berpotensi mengandalkan konteks (misalnya warna air) "
                "dan perlu data atau augmentasi yang lebih beragam.")

    st.subheader("Contoh prediksi pada data uji")
    show_png("prediksi_contoh.png")
    ps = E["pred_sample_summary"]
    st.markdown(f"**Analisis:** Dari {ps['n_shown']} gambar uji acak yang ditampilkan, {ps['n_correct']} diprediksi benar "
                f"({ps['n_correct'] / ps['n_shown']:.0%}). Judul hijau = benar, merah = salah. Persentase di dalam kurung "
                "adalah tingkat keyakinan model.")
    if (ARTIFACT_DIR / "prediksi_salah.png").exists():
        st.subheader("Contoh gambar yang salah diprediksi")
        show_png("prediksi_salah.png")
        st.markdown("**Analisis:** Kesalahan biasanya muncul pada gambar dengan objek kecil, tertutup, buram, atau "
                    "memiliki kemiripan visual dengan kelas lain. Pola ini sejalan dengan pasangan kelas yang sering "
                    "tertukar pada confusion matrix.")

# ---- Tab 5
with tab5:
    st.subheader("Uji prediksi gambar baru")
    st.caption("Unggah gambar organisme laut, prediksi muncul otomatis.")
    up = st.file_uploader("Pilih gambar (JPG/PNG/BMP/WEBP)", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        try:
            pil = Image.open(up).convert("RGB")
        except Exception:
            st.error("File tidak dapat dibaca sebagai gambar. Coba file lain.")
            st.stop()

        img = preprocess(pil)
        probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
        top = int(np.argmax(probs))
        heat = make_gradcam(img, top)

        st.success(f"Hasil prediksi: **{class_names[top]}** (keyakinan {probs[top]:.1%})")
        c1, c2 = st.columns(2)
        c1.image(pil, caption="Gambar yang diunggah", width="stretch")
        c2.image(overlay(img, heat), caption="Grad-CAM: area yang memengaruhi prediksi", width="stretch")

        st.markdown("**Probabilitas tiap kelas**")
        order = np.argsort(probs)
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(class_names) + 1))
        colors = ["#2ecc71" if i == top else "#95a5a6" for i in order]
        ax.barh([class_names[i] for i in order], probs[order], color=colors)
        ax.set_xlim(0, 1); ax.set_xlabel("Probabilitas")
        for y, i in enumerate(order):
            ax.text(probs[i] + 0.01, y, f"{probs[i]:.1%}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Analisis:** " + analisis_prediksi(probs))
        st.markdown("**Analisis Grad-CAM:** Area berwarna merah-kuning adalah bagian gambar yang paling dipakai model "
                    "untuk menentukan kelas. Bila area tersebut menutupi objek organisme, prediksi lebih dapat dipercaya.")
    else:
        st.info("Belum ada gambar yang diunggah.")