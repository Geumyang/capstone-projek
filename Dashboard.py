import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────
# DATA 15 KELAS MAKANAN (TKPI 2017)
# ─────────────────────────────────────────
FOODS = [
    {"key":"bakso",       "name":"Bakso",          "kal":76,  "protein":4.1,  "lemak":2.5,  "karbo":9.2,  "sumber":"TKPI_2017"},
    {"key":"bubur_ayam",  "name":"Bubur Ayam",     "kal":80,  "protein":4.8,  "lemak":2.2,  "karbo":10.5, "sumber":"TKPI_komposit"},
    {"key":"gado_gado",   "name":"Gado-gado",      "kal":137, "protein":6.1,  "lemak":3.2,  "karbo":21.0, "sumber":"TKPI_2017"},
    {"key":"klepon",      "name":"Klepon",         "kal":177, "protein":2.1,  "lemak":2.8,  "karbo":36.4, "sumber":"TKPI_2017"},
    {"key":"mie_goreng",  "name":"Mie Goreng Jawa","kal":468, "protein":7.6,  "lemak":20.4, "karbo":62.4, "sumber":"TKPI_2017"},
    {"key":"nasi_goreng", "name":"Nasi Goreng",    "kal":276, "protein":3.2,  "lemak":3.2,  "karbo":30.2, "sumber":"TKPI_2017"},
    {"key":"nasi_gudeg",  "name":"Nasi Gudeg",     "kal":165, "protein":5.8,  "lemak":6.2,  "karbo":22.5, "sumber":"TKPI_komposit"},
    {"key":"nasi_kuning", "name":"Nasi Kuning",    "kal":158, "protein":3.2,  "lemak":3.5,  "karbo":29.8, "sumber":"TKPI_komposit"},
    {"key":"nasi_padang", "name":"Nasi Padang",    "kal":210, "protein":9.5,  "lemak":8.0,  "karbo":27.0, "sumber":"TKPI_komposit"},
    {"key":"pempek",      "name":"Pempek",         "kal":157, "protein":4.5,  "lemak":2.2,  "karbo":29.2, "sumber":"TKPI_2017"},
    {"key":"rawon",       "name":"Rawon",          "kal":60,  "protein":5.4,  "lemak":2.5,  "karbo":4.0,  "sumber":"TKPI_2017"},
    {"key":"rendang",     "name":"Rendang",        "kal":193, "protein":22.6, "lemak":7.9,  "karbo":7.8,  "sumber":"TKPI_2017"},
    {"key":"sate_ayam",   "name":"Sate Ayam",      "kal":170, "protein":17.5, "lemak":9.2,  "karbo":4.8,  "sumber":"TKPI_2017"},
    {"key":"soto",        "name":"Soto",           "kal":96,  "protein":3.4,  "lemak":6.7,  "karbo":5.8,  "sumber":"TKPI_2017"},
    {"key":"tahu_gejrot", "name":"Tahu Gejrot",    "kal":102, "protein":6.8,  "lemak":5.1,  "karbo":8.2,  "sumber":"TKPI_komposit"},
]
df_foods = pd.DataFrame(FOODS)
df_foods["kategori_kalori"] = df_foods["kal"].apply(
    lambda k: "Rendah" if k < 100 else ("Sedang" if k < 200 else "Tinggi")
)

# ─────────────────────────────────────────
# DATA PERFORMA MODEL YOLOV8
# ─────────────────────────────────────────
df_model = pd.DataFrame({
    "Kelas":     ["bakso","bubur ayam","gado-gado","klepon","mie goreng jawa",
                  "nasi goreng","nasi gudeg","nasi kuning","nasi padang","pempek",
                  "rawon","rendang","sate ayam","soto","tahu gejrot"],
    "Images":    [92,60,71,63,56,54,32,31,57,60,51,55,55,60,56],
    "Instances": [188,71,71,78,57,54,32,31,58,62,54,55,58,61,58],
    "Precision": [0.957,0.971,0.997,0.978,0.977,1.000,0.994,0.994,0.996,
                  0.950,0.998,0.979,1.000,0.994,0.995],
    "Recall":    [0.833,0.959,0.986,0.962,1.000,0.995,1.000,1.000,0.983,
                  0.952,1.000,0.982,1.000,1.000,1.000],
    "mAP50":     [0.947,0.983,0.995,0.993,0.994,0.995,0.995,0.995,0.985,
                  0.946,0.995,0.992,0.995,0.995,0.995],
    "mAP50_95":  [0.727,0.906,0.936,0.827,0.935,0.954,0.958,0.949,0.962,
                  0.866,0.826,0.949,0.908,0.942,0.942],
})

# ─────────────────────────────────────────
# LOAD AKG DARI EXCEL
# ─────────────────────────────────────────
@st.cache_data
def load_akg():
    return pd.read_excel("dataset_AKG_Lengkap.xlsx")

def get_akg(df_akg, usia_str, gender_int, kondisi):
    row = df_akg[df_akg["usia"] == usia_str]
    row = row[row["gender"] == gender_int]
    if gender_int == 1:
        if kondisi == "Normal":
            row = row[(row["hamil_1_13"]==0)&(row["hamil_14_27"]==0)&
                      (row["hamil_28_41"]==0)&(row["menyusui_6bl_pertama"]==0)&
                      (row["menyusui_6bl_kedua"]==0)]
        elif kondisi == "Hamil trimester 1":   row = row[row["hamil_1_13"]==1]
        elif kondisi == "Hamil trimester 2":   row = row[row["hamil_14_27"]==1]
        elif kondisi == "Hamil trimester 3":   row = row[row["hamil_28_41"]==1]
        elif kondisi == "Menyusui 0-6 bulan":  row = row[row["menyusui_6bl_pertama"]==1]
        elif kondisi == "Menyusui 7-12 bulan": row = row[row["menyusui_6bl_kedua"]==1]
    if row.empty: return None
    r = row.iloc[0]
    return {"kal":int(r["energi_kkal"]), "protein":int(r["protein_g"]),
            "lemak":float(r["lemak_total_g"]), "karbo":int(r["karbohidrat_g"])}

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(page_title="NutriCitra", layout="wide")
st.title("NutriCitra — Analisis Gizi Makanan Indonesia")
st.caption(
    "Dashboard analitik berbasis model deteksi objek YOLOv8 | "
    "Data nutrisi: TKPI 2017 (Kemenkes RI) | "
    "Referensi AKG: Permenkes No. 28 Tahun 2019"
)

df_akg = load_akg()

# ─────────────────────────────────────────
# SIDEBAR — PROFIL PENGGUNA
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Profil Pengguna")
    st.caption("Profil digunakan untuk menyesuaikan nilai AKG yang menjadi acuan perhitungan.")
    gender     = st.radio("Jenis kelamin:", ["Laki-laki", "Perempuan"])
    gender_int = 0 if gender == "Laki-laki" else 1
    usia       = st.slider("Usia (tahun):", 1, 80, 25)
    usia_str = "80+ tahun" if usia == 80 else f"{usia} tahun"
    kondisi    = "Normal"
    if gender == "Perempuan":
        kondisi = st.selectbox("Kondisi fisiologis:", [
            "Normal", "Hamil trimester 1", "Hamil trimester 2",
            "Hamil trimester 3", "Menyusui 0-6 bulan", "Menyusui 7-12 bulan"
        ])
    akg = get_akg(df_akg, usia_str, gender_int, kondisi)
    if akg:
        st.divider()
        st.subheader("AKG Harian")
        c1, c2 = st.columns(2)
        c1.metric("Kalori",      f"{akg['kal']} kkal")
        c1.metric("Lemak",       f"{akg['lemak']} g")
        c2.metric("Protein",     f"{akg['protein']} g")
        c2.metric("Karbohidrat", f"{akg['karbo']} g")
        st.caption("Sumber: Permenkes No. 28 Tahun 2019")
    else:
        st.warning("Data AKG tidak ditemukan untuk profil yang dipilih.")

# ─────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Kalkulator Nutrisi",
    "Eksplorasi Makanan",
    "Performa Model",
])

# ════════════════════════════════════════
# TAB 1 — KALKULATOR NUTRISI
# Pertanyaan Bisnis 5, 6, 7
# ════════════════════════════════════════
with tab1:
    st.subheader("Kalkulator Kontribusi Nutrisi per Porsi")
    st.markdown("""
    **Pertanyaan Bisnis:**
    Seberapa besar kontribusi satu porsi makanan Indonesia terhadap pemenuhan
    kebutuhan energi dan makronutrien harian berdasarkan Angka Kecukupan Gizi (AKG)
    yang disesuaikan dengan profil pengguna?
    """)
    st.divider()

    cf1, cf2 = st.columns(2)
    with cf1:
        kat_filter = st.radio(
            "Filter berdasarkan kategori kalori:",
            ["Semua", "Rendah (<100 kkal)", "Sedang (100-199 kkal)", "Tinggi (>=200 kkal)"],
            horizontal=True
        )
    with cf2:
        sumber_filter = st.radio(
            "Filter berdasarkan sumber data:",
            ["Semua", "TKPI_2017", "TKPI_komposit"],
            horizontal=True
        )

    df_f = df_foods.copy()
    if kat_filter == "Rendah (<100 kkal)":
        df_f = df_f[df_f["kal"] < 100]
    elif kat_filter == "Sedang (100-199 kkal)":
        df_f = df_f[(df_f["kal"] >= 100) & (df_f["kal"] < 200)]
    elif kat_filter == "Tinggi (>=200 kkal)":
        df_f = df_f[df_f["kal"] >= 200]
    if sumber_filter != "Semua":
        df_f = df_f[df_f["sumber"] == sumber_filter]

    if df_f.empty:
        st.info("Tidak ada data makanan yang sesuai dengan filter yang dipilih.")
    else:
        col_sel, col_berat = st.columns(2)
        with col_sel:
            makanan_sel = st.selectbox("Pilih makanan:", df_f["name"].tolist())
        with col_berat:
            berat = st.slider("Estimasi berat porsi (gram):", 50, 500, 200, 10)

        food    = df_f[df_f["name"] == makanan_sel].iloc[0]
        fk      = berat / 100
        kal_p   = round(food["kal"]     * fk)
        prot_p  = round(food["protein"] * fk, 1)
        lemak_p = round(food["lemak"]   * fk, 1)
        karbo_p = round(food["karbo"]   * fk, 1)

        st.divider()

        if akg:
            pct_kal   = round(kal_p   / akg["kal"]     * 100, 1)
            pct_prot  = round(prot_p  / akg["protein"]  * 100, 1)
            pct_lemak = round(lemak_p / akg["lemak"]    * 100, 1)
            pct_karbo = round(karbo_p / akg["karbo"]    * 100, 1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Kalori",      f"{kal_p} kkal",   f"{pct_kal}% AKG")
            m2.metric("Protein",     f"{prot_p} g",     f"{pct_prot}% AKG")
            m3.metric("Lemak",       f"{lemak_p} g",    f"{pct_lemak}% AKG")
            m4.metric("Karbohidrat", f"{karbo_p} g",    f"{pct_karbo}% AKG")

            df_chart = pd.DataFrame({
                "Nutrisi": ["Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)"],
                "% AKG":   [pct_kal, pct_prot, pct_lemak, pct_karbo]
            })
            fig = px.bar(
                df_chart, x="Nutrisi", y="% AKG",
                color="% AKG",
                color_continuous_scale=["#3B6D11", "#EF9F27", "#A32D2D"],
                range_color=[0, 50],
                text="% AKG",
                title=(
                    f"Kontribusi terhadap AKG — {berat} g {makanan_sel} "
                    f"({gender}, {usia} tahun, {kondisi})"
                )
            )
            fig.add_hline(
                y=25, line_dash="dash", line_color="orange",
                annotation_text="Batas 25% AKG per porsi makan"
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

            # Interpretasi dan implikasi — di bawah grafik
            st.markdown("**Interpretasi dan Implikasi**")

            nutrisi_tinggi = [
                n for n, p in zip(
                    ["kalori", "protein", "lemak", "karbohidrat"],
                    [pct_kal, pct_prot, pct_lemak, pct_karbo]
                ) if p >= 25
            ]
            nutrisi_rendah = [
                n for n, p in zip(
                    ["kalori", "protein", "lemak", "karbohidrat"],
                    [pct_kal, pct_prot, pct_lemak, pct_karbo]
                ) if p < 10
            ]

            if pct_kal >= 25:
                st.warning(
                    f"Konsumsi {berat} g {makanan_sel} memenuhi {pct_kal}% kebutuhan energi harian "
                    f"untuk {gender.lower()} usia {usia} tahun ({kondisi.lower()}). "
                    f"Nilai ini melampaui batas 25% yang secara umum direkomendasikan per satu kali makan. "
                    f"Perlu dipertimbangkan apabila makanan ini dikonsumsi bersamaan dengan hidangan lain."
                )
            else:
                st.success(
                    f"Konsumsi {berat} g {makanan_sel} memenuhi {pct_kal}% kebutuhan energi harian "
                    f"untuk {gender.lower()} usia {usia} tahun ({kondisi.lower()}). "
                    f"Kontribusi kalori ini masih berada di bawah batas 25% per porsi makan."
                )
            if nutrisi_tinggi:
                st.info(
                    f"Kandungan {', '.join(nutrisi_tinggi)} dari porsi ini tergolong relatif tinggi "
                    f"(kontribusi ≥25% AKG). Hal ini perlu dipertimbangkan dalam perencanaan menu harian."
                )
            if nutrisi_rendah:
                st.info(
                    f"Kandungan {', '.join(nutrisi_rendah)} dari porsi ini tergolong rendah "
                    f"(kontribusi <10% AKG). Diperlukan sumber makanan tambahan untuk memenuhi kebutuhan harian."
                )

        else:
            st.warning("Profil pengguna belum lengkap. Lengkapi data pada panel sebelah kiri untuk melihat persentase AKG.")

        st.caption(
            f"Sumber data nutrisi: "
            f"{'TKPI 2017 (Kemenkes RI)' if food['sumber'] == 'TKPI_2017' else 'TKPI 2017 — data komposit (lihat keterangan di bagian akhir dashboard)'}"
        )

# ════════════════════════════════════════
# TAB 2 — EKSPLORASI MAKANAN
# Pertanyaan Bisnis 1, 2, 3, 4
# ════════════════════════════════════════
with tab2:
    st.subheader("Analisis Profil Nutrisi 15 Makanan Indonesia")
    st.markdown("""
    Bagian ini menyajikan analisis komparatif kandungan gizi 15 makanan Indonesia
    yang tercakup dalam sistem NutriCitra, berdasarkan data Tabel Komposisi Pangan
    Indonesia (TKPI) 2017.
    """)

    # ── Pertanyaan Bisnis 3: Distribusi kategori kalori ──
    st.markdown("#### Pertanyaan Bisnis 3 — Distribusi Kategori Kalori")
    st.markdown(
        "Bagaimana distribusi 15 makanan yang dapat dideteksi sistem "
        "berdasarkan kategori tingkat kalorinya per 100 gram?"
    )

    kat_count = df_foods["kategori_kalori"].value_counts().reset_index()
    kat_count.columns = ["Kategori", "Jumlah"]
    fig_pie = px.pie(
        kat_count, names="Kategori", values="Jumlah",
        color="Kategori",
        color_discrete_map={"Rendah": "#3B6D11", "Sedang": "#854F0B", "Tinggi": "#A32D2D"},
        title="Distribusi 15 makanan berdasarkan kategori kalori (per 100 g)"
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Temuan**")
    st.info(
        "Mayoritas dari 15 makanan yang dianalisis termasuk dalam kategori kalori sedang "
        "(100-199 kkal per 100 g). Makanan berkategori rendah kalori, seperti rawon, soto, "
        "dan bakso, umumnya berbasis kuah dengan kandungan lemak yang lebih rendah. "
        "Sementara itu, mie goreng jawa dan nasi goreng masuk kategori tinggi kalori karena "
        "proses pengolahan dengan minyak serta dominasi karbohidrat dalam komposisinya."
    )
    st.markdown("**Implikasi**")
    st.info(
        "Distribusi ini menunjukkan bahwa sebagian besar makanan tradisional Indonesia "
        "yang terdeteksi sistem berada pada rentang kalori yang moderat. "
        "Pemilihan jenis makanan dan estimasi berat porsi menjadi faktor penting "
        "dalam pengelolaan asupan energi harian."
    )

    st.divider()

    # ── Pertanyaan Bisnis 1 & 2: Ranking nutrisi ──
    st.markdown("#### Pertanyaan Bisnis 1 & 2 — Perbandingan Kandungan Nutrisi")
    st.markdown(
        "Makanan mana yang memiliki kandungan kalori dan protein tertinggi "
        "serta terendah per 100 gram di antara 15 makanan yang dianalisis?"
    )

    cr1, cr2 = st.columns(2)
    with cr1:
        rank_by = st.selectbox(
            "Urutkan berdasarkan:",
            ["kal", "protein", "lemak", "karbo"],
            format_func=lambda x: {
                "kal": "Kalori (kkal)", "protein": "Protein (g)",
                "lemak": "Lemak (g)", "karbo": "Karbohidrat (g)"
            }[x]
        )
    with cr2:
        kat_exp = st.multiselect(
            "Filter kategori kalori:",
            ["Rendah", "Sedang", "Tinggi"],
            default=["Rendah", "Sedang", "Tinggi"]
        )

    df_exp  = df_foods[df_foods["kategori_kalori"].isin(kat_exp)].sort_values(rank_by, ascending=False)
    unit    = {"kal": "kkal", "protein": "g", "lemak": "g", "karbo": "g"}[rank_by]
    label   = {"kal": "Kalori", "protein": "Protein", "lemak": "Lemak", "karbo": "Karbohidrat"}[rank_by]

    if df_exp.empty:
        st.info("Tidak ada data yang sesuai dengan filter kategori yang dipilih.")
    else:
        fig2 = px.bar(
            df_exp, x=rank_by, y="name", orientation="h",
            color="kategori_kalori",
            color_discrete_map={"Rendah": "#3B6D11", "Sedang": "#854F0B", "Tinggi": "#A32D2D"},
            text=rank_by,
            labels={
                "name": "Makanan",
                rank_by: f"{label} ({unit})",
                "kategori_kalori": "Kategori Kalori"
            },
            title=f"Perbandingan kandungan {label.lower()} per 100 g"
        )
        fig2.update_traces(texttemplate=f"%{{text}} {unit}", textposition="outside")
        fig2.update_layout(height=480, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

        top1  = df_exp.iloc[0]
        last1 = df_exp.iloc[-1]
        st.markdown("**Temuan**")
        st.info(
            f"Berdasarkan data TKPI 2017, makanan dengan kandungan {label.lower()} tertinggi "
            f"per 100 g adalah {top1['name']} ({top1[rank_by]} {unit}), "
            f"sedangkan yang terendah adalah {last1['name']} ({last1[rank_by]} {unit}). "
            f"Perbedaan nilai ini mencerminkan variasi komposisi bahan dan metode pengolahan "
            f"yang berbeda antar makanan."
        )
        st.markdown("**Implikasi**")
        st.info(
            f"Perbedaan kandungan {label.lower()} yang signifikan antar makanan menunjukkan "
            f"pentingnya informasi komposisi gizi dalam perencanaan pola makan. "
            f"Sistem NutriCitra dapat membantu pengguna memperoleh informasi ini secara otomatis "
            f"melalui deteksi makanan berbasis gambar."
        )

    st.divider()

    # ── Pertanyaan Bisnis 4: Efisiensi protein ──
    st.markdown("#### Pertanyaan Bisnis 4 — Indeks Efisiensi Protein")
    st.markdown(
        "Makanan mana yang memberikan kandungan protein tertinggi "
        "dengan kalori yang relatif rendah?"
    )

    df_eff = df_foods.copy()
    df_eff["efisiensi"] = (df_eff["protein"] / df_eff["kal"] * 100).round(2)
    df_eff = df_eff.sort_values("efisiensi", ascending=False)

    fig3 = px.bar(
        df_eff, x="efisiensi", y="name", orientation="h",
        color="efisiensi",
        color_continuous_scale=["#EF9F27", "#3B6D11"],
        text="efisiensi",
        labels={"name": "Makanan", "efisiensi": "Gram protein per 100 kkal"},
        title="Indeks efisiensi protein — gram protein per 100 kkal (sumber: TKPI 2017)"
    )
    fig3.update_traces(texttemplate="%{text:.2f} g/100 kkal", textposition="outside")
    fig3.update_layout(height=480, coloraxis_showscale=False,
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    top_eff  = df_eff.iloc[0]
    last_eff = df_eff.iloc[-1]
    st.markdown("**Temuan**")
    st.info(
        f"{top_eff['name']} memiliki indeks efisiensi protein tertinggi sebesar "
        f"{top_eff['efisiensi']:.2f} g protein per 100 kkal, yang mencerminkan kandungan protein "
        f"yang relatif tinggi dengan kalori yang rendah. Sebaliknya, {last_eff['name']} "
        f"menunjukkan indeks efisiensi terendah ({last_eff['efisiensi']:.2f} g/100 kkal), "
        f"yang disebabkan oleh dominasi karbohidrat dalam komposisinya."
    )
    st.markdown("**Implikasi**")
    st.info(
        "Indeks efisiensi protein dapat menjadi acuan dalam memilih makanan yang mendukung "
        "pemenuhan kebutuhan protein tanpa kelebihan asupan energi, terutama bagi kelompok "
        "yang memerlukan manajemen kalori secara ketat."
    )

    st.divider()

    # ── Ringkasan temuan tab 2 ──
    st.markdown("#### Ringkasan Temuan Analisis Nutrisi")
    st.success("""
    Berdasarkan analisis komparatif data TKPI 2017 terhadap 15 makanan Indonesia yang tercakup
    dalam sistem NutriCitra, diperoleh temuan sebagai berikut.

    Pertama, rendang dan sate ayam tercatat sebagai makanan dengan kandungan protein tertinggi
    per 100 gram di antara seluruh kelas yang dianalisis.

    Kedua, mie goreng jawa memiliki kandungan kalori tertinggi (468 kkal per 100 g),
    yang perlu diperhatikan dalam konteks manajemen asupan energi harian.

    Ketiga, rawon menunjukkan indeks efisiensi protein tertinggi, menjadikannya pilihan
    yang relevan bagi pengguna yang ingin mengoptimalkan asupan protein dengan kalori rendah.

    Keempat, sebagian besar makanan dalam dataset berada pada kategori kalori sedang
    (100-199 kkal per 100 g), yang mencerminkan profil gizi makanan tradisional Indonesia secara umum.
    """)

# ════════════════════════════════════════
# TAB 3 — PERFORMA MODEL
# ════════════════════════════════════════
with tab3:
    st.subheader("Evaluasi Performa Model YOLOv8")
    st.markdown("""
    **Pertanyaan Bisnis:**
    Seberapa akurat model YOLOv8 dalam mendeteksi 15 kelas makanan Indonesia,
    dan faktor apa yang memengaruhi tingkat akurasi deteksi per kelas?
    """)

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Total kelas",     "15")
    pm2.metric("Total gambar",    "4.632")
    pm3.metric("mAP50 rata-rata", "0.987")
    pm4.metric("Precision",       "0.985")
    pm5.metric("Recall",          "0.977")

    st.markdown("**Ringkasan Performa**")
    st.success(
        "Model YOLOv8 yang dilatih pada dataset 15 kelas makanan Indonesia mencapai nilai "
        "rata-rata mAP50 sebesar 0.987. Nilai ini menunjukkan bahwa model mampu mendeteksi "
        "objek makanan dengan tingkat akurasi yang tinggi. Sebagian besar kelas memiliki "
        "nilai precision dan recall di atas 0.95, yang mengindikasikan konsistensi deteksi "
        "pada berbagai kondisi gambar."
    )

    st.divider()

    # ── Visualisasi training dari Tim AI ──
    st.markdown("#### Visualisasi Hasil Pelatihan Model")
    st.markdown(
        "Visualisasi berikut dihasilkan selama proses pelatihan model YOLOv8 "
        "dan mencerminkan karakteristik performa model secara keseluruhan."
    )

    IMAGE_FILES = {
        "confusion_matrix_normalized.png": "Confusion Matrix — Sebaran prediksi per kelas",
        "results.png":                     "Grafik Pelatihan — Loss, Precision, Recall, mAP per epoch",
        "BoxPR_curve.png":                 "Precision-Recall Curve — Hubungan presisi dan sensitivitas",
        "BoxF1_curve.png":                 "F1 Curve — Keseimbangan precision dan recall",
    }
    IMG_FOLDER = Path("images")

    if not IMG_FOLDER.exists():
        st.info(
            "Folder 'images/' belum ditemukan. Buat folder bernama 'images' "
            "di direktori yang sama dengan file Dashboard.py, kemudian tempatkan "
            "file gambar hasil ekspor dari proses pelatihan model."
        )
    else:
        found = {k: v for k, v in IMAGE_FILES.items() if (IMG_FOLDER / k).exists()}
        if not found:
            st.info("Folder 'images/' ditemukan, namun belum ada file gambar yang sesuai.")
        else:
            img_sel = st.selectbox("Pilih visualisasi:", list(found.values()))
            img_key = [k for k, v in found.items() if v == img_sel][0]
            st.image(str(IMG_FOLDER / img_key), caption=img_sel, use_container_width=True)

            insights = {
                "Confusion Matrix — Sebaran prediksi per kelas": (
                    "Diagonal utama yang dominan menunjukkan bahwa model secara konsisten "
                    "memprediksi kelas yang benar. Kesalahan klasifikasi yang paling sering "
                    "terjadi pada kelas bakso dan pempek, yang diduga disebabkan oleh "
                    "kemiripan karakteristik visual antara kedua kelas tersebut, terutama "
                    "pada variasi sudut pandang dan kondisi pencahayaan gambar."
                ),
                "Grafik Pelatihan — Loss, Precision, Recall, mAP per epoch": (
                    "Nilai loss yang menurun secara konsisten dan metrik evaluasi yang meningkat "
                    "sepanjang proses pelatihan mengindikasikan bahwa model berhasil mempelajari "
                    "pola data tanpa mengalami overfitting. Konvergensi model tercapai sebelum "
                    "epoch ke-50, menunjukkan efisiensi proses pelatihan."
                ),
                "Precision-Recall Curve — Hubungan presisi dan sensitivitas": (
                    "Kurva PR yang mendekati pojok kanan atas diagram mengindikasikan bahwa model "
                    "mampu mempertahankan nilai precision yang tinggi meskipun nilai recall "
                    "ditingkatkan. Luas area di bawah kurva (AUC) yang besar mengkonfirmasi "
                    "stabilitas performa model pada berbagai nilai threshold."
                ),
                "F1 Curve — Keseimbangan precision dan recall": (
                    "Nilai F1 score yang tinggi pada berbagai nilai threshold menunjukkan "
                    "bahwa model memiliki keseimbangan yang baik antara precision dan recall. "
                    "Karakteristik ini penting dalam konteks aplikasi deteksi makanan, "
                    "di mana kesalahan deteksi dari kedua arah (false positive dan false negative) "
                    "perlu diminimalkan."
                ),
            }
            if img_sel in insights:
                st.markdown("**Interpretasi**")
                st.info(insights[img_sel])

    st.divider()

    # ── Performa per kelas ──
    st.markdown("#### Perbandingan Metrik Evaluasi per Kelas")
    st.markdown(
        "Visualisasi berikut menampilkan distribusi nilai metrik evaluasi "
        "di antara 15 kelas makanan yang dideteksi model."
    )

    pf1, pf2 = st.columns(2)
    with pf1:
        metrik_sel = st.selectbox(
            "Pilih metrik evaluasi:",
            ["mAP50", "mAP50_95", "Precision", "Recall"]
        )
    with pf2:
        sort_dir = st.radio("Urutan tampilan:", ["Tertinggi", "Terendah"], horizontal=True)

    df_m = df_model.sort_values(metrik_sel, ascending=(sort_dir == "Terendah"))
    fig4 = px.bar(
        df_m, x=metrik_sel, y="Kelas", orientation="h",
        text=metrik_sel, color=metrik_sel,
        color_continuous_scale=["#A32D2D", "#EF9F27", "#3B6D11"],
        range_color=[0.92, 1.0],
        title=f"Nilai {metrik_sel} per kelas makanan"
    )
    fig4.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig4.update_layout(height=500, coloraxis_showscale=False,
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

    best  = df_model.loc[df_model["mAP50"].idxmax()]
    worst = df_model.loc[df_model["mAP50"].idxmin()]
    st.markdown("**Temuan**")
    st.info(
        f"Kelas dengan nilai mAP50 tertinggi adalah {best['Kelas']} ({best['mAP50']:.3f}), "
        f"sedangkan kelas dengan nilai terendah adalah {worst['Kelas']} ({worst['mAP50']:.3f}). "
        f"Selisih nilai ini mengindikasikan adanya perbedaan tingkat kesulitan deteksi "
        f"yang dipengaruhi oleh karakteristik visual masing-masing kelas."
    )
    st.markdown("**Implikasi**")
    st.info(
        f"Kelas {worst['Kelas']} berpotensi menjadi prioritas peningkatan pada iterasi "
        f"pengembangan model berikutnya, baik melalui penambahan data pelatihan, "
        f"augmentasi yang lebih variatif, maupun penyesuaian arsitektur model."
    )

    st.divider()

    # ── Jumlah gambar vs akurasi ──
    st.markdown("#### Hubungan antara Volume Data Pelatihan dan Akurasi Deteksi")
    st.markdown(
        "Analisis ini mengevaluasi apakah penambahan jumlah gambar pelatihan "
        "secara langsung berkontribusi pada peningkatan nilai mAP50 per kelas."
    )

    fig5 = px.scatter(
        df_model, x="Images", y="mAP50",
        text="Kelas", size="Images",
        color="mAP50",
        color_continuous_scale=["#A32D2D", "#EF9F27", "#3B6D11"],
        range_color=[0.94, 1.0],
        title="Jumlah gambar pelatihan vs nilai mAP50 per kelas",
        height=420
    )
    fig5.update_traces(textposition="top center")
    fig5.update_layout(coloraxis_colorbar_title="mAP50")
    st.plotly_chart(fig5, use_container_width=True)

    corr = df_model["Images"].corr(df_model["mAP50"])
    kuat = "lemah" if abs(corr) < 0.3 else ("sedang" if abs(corr) < 0.6 else "kuat")
    st.markdown("**Temuan**")
    st.info(
        f"Korelasi antara jumlah gambar pelatihan dan nilai mAP50 adalah {corr:.3f}, "
        f"yang mengindikasikan hubungan yang {kuat}. "
        f"Sebagai contoh, kelas {worst['Kelas']} memiliki jumlah gambar pelatihan "
        f"terbanyak ({int(worst['Images'])} gambar) namun mencatat nilai mAP50 terendah. "
        f"Hal ini menunjukkan bahwa volume data pelatihan bukan satu-satunya faktor penentu "
        f"akurasi deteksi."
    )
    st.markdown("**Implikasi**")
    st.info(
        "Keragaman dan kualitas visual gambar pelatihan, seperti variasi sudut pandang, "
        "pencahayaan, dan latar belakang, kemungkinan besar memiliki pengaruh yang lebih "
        "signifikan terhadap performa model dibandingkan sekadar peningkatan jumlah data."
    )

    st.divider()

    # ── Precision vs Recall ──
    st.markdown("#### Analisis Precision-Recall per Kelas")
    st.markdown(
        "Grafik berikut menggambarkan posisi setiap kelas makanan "
        "dalam ruang precision-recall, dengan ukuran titik yang mencerminkan "
        "jumlah gambar pelatihan."
    )

    fig6 = px.scatter(
        df_model, x="Precision", y="Recall",
        size="Images", color="mAP50", text="Kelas",
        color_continuous_scale=["#A32D2D", "#EF9F27", "#3B6D11"],
        range_color=[0.94, 1.0],
        title="Sebaran nilai precision dan recall per kelas (ukuran titik: jumlah gambar pelatihan)",
        height=450
    )
    fig6.update_traces(textposition="top center")
    fig6.add_hline(y=0.95, line_dash="dot", line_color="gray",
                   annotation_text="Recall = 0.95", annotation_position="bottom right")
    fig6.add_vline(x=0.95, line_dash="dot", line_color="gray",
                   annotation_text="Precision = 0.95")
    st.plotly_chart(fig6, use_container_width=True)

    below_recall = df_model[df_model["Recall"] < 0.95]["Kelas"].tolist()
    below_prec   = df_model[df_model["Precision"] < 0.95]["Kelas"].tolist()

    st.markdown("**Temuan**")
    temuan_pr = "Mayoritas kelas berada pada zona ideal dengan precision dan recall di atas 0.95. "
    if below_recall:
        temuan_pr += f"Kelas yang masih berada di bawah threshold recall 0.95 adalah {', '.join(below_recall)}. "
    if below_prec:
        temuan_pr += f"Kelas yang masih berada di bawah threshold precision 0.95 adalah {', '.join(below_prec)}. "
    st.info(temuan_pr)

    st.markdown("**Implikasi**")
    st.info(
        "Kelas-kelas yang berada di luar zona ideal memerlukan perhatian lebih lanjut, "
        "baik dari sisi kualitas data pelatihan maupun strategi augmentasi. "
        "Peningkatan recall pada kelas tertentu dapat diprioritaskan apabila aplikasi "
        "menuntut sensitivitas deteksi yang tinggi."
    )

    with st.expander("Lihat tabel lengkap metrik evaluasi per kelas"):
        st.dataframe(df_model, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# KESIMPULAN AKHIR
# ════════════════════════════════════════
st.divider()
st.markdown("### Kesimpulan")
st.success("""
Berdasarkan seluruh analisis yang disajikan dalam dashboard ini, diperoleh enam kesimpulan utama.

Pertama, integrasi data TKPI 2017 dan AKG Permenkes No. 28 Tahun 2019 memungkinkan estimasi
kontribusi nutrisi makanan terhadap kebutuhan gizi harian secara personal dan terukur.

Kedua, rendang dan sate ayam tercatat sebagai makanan dengan kandungan protein tertinggi
di antara 15 kelas yang dianalisis, sedangkan mie goreng jawa memiliki kandungan kalori
tertinggi yang perlu mendapat perhatian dalam konteks manajemen asupan energi.

Ketiga, rawon menunjukkan indeks efisiensi protein tertinggi, menjadikannya relevan
sebagai pilihan makanan bagi kelompok yang membutuhkan asupan protein dengan kalori rendah.

Keempat, model YOLOv8 mencapai nilai rata-rata mAP50 sebesar 0.987 pada dataset
15 kelas makanan Indonesia, yang menunjukkan performa deteksi yang tinggi dan konsisten.

Kelima, analisis korelasi menunjukkan bahwa keragaman visual gambar pelatihan lebih
berpengaruh terhadap akurasi deteksi dibandingkan volume data semata, sebagaimana
tercermin dari kasus kelas dengan jumlah gambar terbanyak namun nilai mAP50 terendah.

Keenam, sistem NutriCitra tidak hanya melakukan klasifikasi makanan, tetapi juga
menerjemahkan hasil deteksi menjadi informasi gizi yang kontekstual berdasarkan profil
AKG pengguna, dimana hal ini merupakan sebuah pendekatan yang membedakan sistem ini dari aplikasi deteksi makanan
pada umumnya.
""")

# ════════════════════════════════════════
# KETERANGAN DATA — BAGIAN AKHIR
# ════════════════════════════════════════
st.divider()
with st.expander("Keterangan Sumber Data dan Metodologi"):
    st.markdown("""
    **TKPI_2017**
    Data diambil langsung dari Tabel Komposisi Pangan Indonesia 2017
    yang diterbitkan oleh Kementerian Kesehatan Republik Indonesia.
    Nilai yang tersedia merupakan hasil analisis laboratorium terhadap
    bahan pangan dalam kondisi matang siap konsumsi.

    **TKPI_komposit**
    Nilai estimasi yang dihitung berdasarkan komponen penyusun makanan
    dengan mengacu pada data TKPI 2017. Pendekatan ini diterapkan pada
    makanan yang tidak memiliki entri tunggal dalam TKPI karena terdiri
    atas beberapa komponen dalam satu hidangan (misalnya nasi padang
    yang terdiri atas nasi, lauk, dan kuah). Nilai yang dihasilkan
    merupakan estimasi berdasarkan proporsi bahan standar dan tidak
    setara dengan data hasil analisis laboratorium langsung.

    Makanan dalam dataset yang menggunakan pendekatan TKPI komposit
    adalah bubur ayam, nasi gudeg, nasi kuning, nasi padang, dan tahu gejrot.

    **Angka Kecukupan Gizi (AKG)**
    Nilai AKG yang digunakan bersumber dari Peraturan Menteri Kesehatan
    Republik Indonesia Nomor 28 Tahun 2019 tentang Angka Kecukupan Gizi
    yang Dianjurkan untuk Masyarakat Indonesia. Nilai AKG disesuaikan
    berdasarkan kelompok usia, jenis kelamin, dan kondisi fisiologis
    pengguna yang dimasukkan melalui panel profil.
    """)

st.caption(
    "Data nutrisi: FatSecret "
    "AKG: Permenkes No. 28 Tahun 2019 | "
    "Model: YOLOv8 — 4.632 gambar pelatihan — 15 kelas makanan Indonesia"
)
