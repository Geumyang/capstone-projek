import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────
# LOAD DATABASE MAKANAN (Tab 1 & Tab 2)
# Baca dari Database.xlsx — 289 makanan Indonesia
# ─────────────────────────────────────────
@st.cache_data
def load_food_database():
    df = pd.read_csv("mastersheet_combined.csv")
    # Hapus baris yang tidak punya data nutrisi
    df = df.dropna(subset=["calories", "fat", "carbohydrate", "protein"])
    # Buat kolom kategori kalori per porsi
    df["kategori_kalori"] = df["calories"].apply(
        lambda k: "Rendah (<200 kkal)"
        if k < 200 else ("Sedang (200–400 kkal)" if k < 400 else "Tinggi (≥400 kkal)")
    )
    # Buat kolom efisiensi protein (per 100 kkal) untuk Tab 2
    df["efisiensi_protein"] = (df["protein"] / df["calories"] * 100).round(2)
    return df

# ─────────────────────────────────────────
# DATA PERFORMA MODEL YOLOV8
# Tetap hardcoded — ini hasil training, bukan database
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
    "Data nutrisi: Database 289 makanan Indonesia | "
    "Referensi AKG: Permenkes No. 28 Tahun 2019"
)

df_akg = load_akg()
df_foods = load_food_database()   # ← sekarang dari Database.xlsx

# ─────────────────────────────────────────
# SIDEBAR — PROFIL PENGGUNA
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Profil Pengguna")
    st.caption("Profil digunakan untuk menyesuaikan nilai AKG yang menjadi acuan perhitungan.")
    gender     = st.radio("Jenis kelamin:", ["Laki-laki", "Perempuan"])
    gender_int = 0 if gender == "Laki-laki" else 1
    usia       = st.slider("Usia (tahun):", 1, 80, 25)
    usia_str   = "80+ tahun" if usia == 80 else f"{usia} tahun"
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
# Data: Database.xlsx (289 makanan, per porsi)
# ════════════════════════════════════════
with tab1:
    st.subheader("Kalkulator Kontribusi Nutrisi per Porsi")
    st.markdown("""

    Makanan tradisional khas Indonesia umumnya tidak memiliki informasi nilai gizi yang mudah diakses.
    Akibatnya, masyarakat sering kali kesulitan memperkirakan apakah suatu makanan sudah memenuhi
    atau bahkan melebihi kebutuhan energi dan zat gizi hariannya.

    Melalui analisis ini, pengguna dapat mengetahui seberapa besar kontribusi satu porsi makanan
    terhadap kebutuhan kalori, protein, lemak, dan karbohidrat berdasarkan profil usia,
    jenis kelamin, serta kondisi fisiologis yang dipilih.
    """)
    st.info(
        f"**{len(df_foods)} makanan tersedia** dari database. "
        "Data nutrisi disajikan **per porsi** sesuai deskripsi sajian pada database.",
        icon="🗄️"
    )
    st.divider()

    # ── Filter ──
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        kat_filter = st.selectbox(
            "Filter kategori kalori per porsi:",
            ["Semua", "Rendah (<200 kkal)", "Sedang (200–400 kkal)", "Tinggi (≥400 kkal)"]
        )
    with cf2:
        daerah_opts = ["Semua"] + sorted(df_foods["asal_daerah"].dropna().unique().tolist())
        daerah_filter = st.selectbox("Filter asal daerah:", daerah_opts)
    with cf3:
        search_kw = st.text_input("Cari nama makanan:", placeholder="Contoh: soto, nasi, ayam…")

    # Terapkan filter
    df_f = df_foods.copy()
    if kat_filter != "Semua":
        df_f = df_f[df_f["kategori_kalori"] == kat_filter]
    if daerah_filter != "Semua":
        df_f = df_f[df_f["asal_daerah"] == daerah_filter]
    if search_kw.strip():
        df_f = df_f[df_f["food_name"].str.contains(search_kw.strip(), case=False, na=False)]

    if df_f.empty:
        st.info("Tidak ada makanan yang sesuai dengan filter yang dipilih.")
    else:
        col_sel, col_info = st.columns([3, 2])
        with col_sel:
            makanan_sel = st.selectbox(
                f"Pilih makanan ({len(df_f)} tersedia):",
                df_f["food_name"].tolist()
            )
        food = df_f[df_f["food_name"] == makanan_sel].iloc[0]

        with col_info:
            st.markdown("**Informasi Porsi Standar**")
            serving_desc = food.get("serving_description", "—")
            serving_g    = food.get("serving_size_g", None)
            if pd.notna(serving_g):
                st.markdown(f"- Sajian: **{serving_desc}** ({int(serving_g)} g)")
            else:
                st.markdown(f"- Sajian: **{serving_desc}**")
            st.markdown(f"- Asal daerah: **{food.get('asal_daerah', '—')}**")

        # ── Slider porsi: gram jika ada serving_size_g, item jika tidak ──
        if pd.notna(serving_g):
            serving_base = int(serving_g)
            berat = st.slider(
                "Estimasi berat porsi (gram):",
                min_value=50,
                max_value=max(800, serving_base * 2),
                value=serving_base,
                step=10,
                help=f"Porsi standar = {serving_base} g. Geser untuk menyesuaikan."
            )
            fk = berat / serving_base          # faktor skala relatif terhadap porsi standar
            label_porsi = f"{berat} g"
        else:
            # Data per item (1 Tusuk, 1 Buah, dll) → slider jumlah
            unit_label = serving_desc.replace("1 ", "").strip() if serving_desc.startswith("1 ") else serving_desc
            jumlah = st.slider(
                f"Jumlah ({unit_label}):",
                min_value=1, max_value=20, value=1, step=1,
                help=f"Satu unit = {serving_desc}"
            )
            fk = jumlah
            label_porsi = f"{jumlah}× {serving_desc}"

        # ── Hitung nilai nutrisi sesuai porsi ──
        kal_p   = round(food["calories"]     * fk)
        prot_p  = round(food["protein"]      * fk, 1)
        lemak_p = round(food["fat"]          * fk, 1)
        karbo_p = round(food["carbohydrate"] * fk, 1)

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
                    f"Kontribusi terhadap AKG — {label_porsi} {makanan_sel} "
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

            # ── Interpretasi ──
            st.markdown("**Interpretasi dan Implikasi**")
            nutrisi_tinggi = [n for n, p in zip(
                ["kalori", "protein", "lemak", "karbohidrat"],
                [pct_kal, pct_prot, pct_lemak, pct_karbo]) if p >= 25]
            nutrisi_rendah = [n for n, p in zip(
                ["kalori", "protein", "lemak", "karbohidrat"],
                [pct_kal, pct_prot, pct_lemak, pct_karbo]) if p < 10]

            if pct_kal >= 25:
                st.warning(
                    f"Konsumsi {label_porsi} {makanan_sel} memenuhi {pct_kal}% kebutuhan energi harian "
                    f"untuk {gender.lower()} usia {usia} tahun ({kondisi.lower()}). "
                    f"Nilai ini melampaui batas 25% per satu kali makan. "
                    f"Perlu dipertimbangkan apabila dikonsumsi bersamaan dengan hidangan lain."
                )
            else:
                st.success(
                    f"Konsumsi {label_porsi} {makanan_sel} memenuhi {pct_kal}% kebutuhan energi harian "
                    f"untuk {gender.lower()} usia {usia} tahun ({kondisi.lower()}). "
                    f"Kontribusi kalori ini masih di bawah batas 25% per porsi makan."
                )
            if nutrisi_tinggi:
                st.info(
                    f"Kandungan {', '.join(nutrisi_tinggi)} dari porsi ini tergolong tinggi "
                    f"(≥25% AKG). Perlu dipertimbangkan dalam perencanaan menu harian."
                )
            if nutrisi_rendah:
                st.info(
                    f"Kandungan {', '.join(nutrisi_rendah)} dari porsi ini tergolong rendah "
                    f"(<10% AKG). Diperlukan sumber makanan tambahan untuk memenuhi kebutuhan harian."
                )

            # ── Serat & Air jika tersedia ──
            extra = []
            if pd.notna(food.get("serat")): extra.append(("Serat",  round(food["serat"] * fk, 1), "g"))
            if pd.notna(food.get("air")):   extra.append(("Air",    round(food["air"]   * fk, 1), "g"))
            if extra:
                st.markdown("**Kandungan Tambahan**")
                ecols = st.columns(len(extra))
                for i, (lbl, val, unit) in enumerate(extra):
                    ecols[i].metric(lbl, f"{val} {unit}")
        else:
            st.warning("Lengkapi profil pengguna di panel kiri untuk melihat persentase AKG.")

        st.caption("Sumber data: Database makanan Indonesia | Data disajikan per porsi.")

# ════════════════════════════════════════
# TAB 2 — EKSPLORASI MAKANAN
# Data: Database.xlsx (289 makanan, per porsi)
# ════════════════════════════════════════
with tab2:
    st.subheader(f"Analisis Profil Nutrisi {len(df_foods)} Makanan Indonesia")
    st.markdown("""

    Di antara ratusan makanan Indonesia yang tersedia dalam database,
    makanan mana yang memiliki kandungan energi, protein, lemak,
    dan karbohidrat paling tinggi atau paling rendah?

    Analisis ini membantu pengguna membandingkan profil nutrisi berbagai makanan
    sehingga dapat memilih menu yang lebih sesuai dengan kebutuhan gizi,
    tujuan diet, maupun kondisi kesehatan tertentu.
""")
    st.divider()

    # ── Distribusi kategori kalori ──
    st.markdown("#### Distribusi Kategori Kalori per Porsi")
    st.markdown("""
    Apakah mayoritas makanan dalam database termasuk makanan rendah,
    sedang, atau tinggi kalori?

    Informasi ini membantu memahami karakteristik umum makanan Indonesia
    serta memberikan gambaran awal mengenai potensi kontribusinya terhadap
    asupan energi harian.
""")

    kat_count = df_foods["kategori_kalori"].value_counts().reset_index()
    kat_count.columns = ["Kategori", "Jumlah"]
    fig_pie = px.pie(
        kat_count, names="Kategori", values="Jumlah",
        color="Kategori",
        color_discrete_map={
            "Rendah (<200 kkal)":    "#3B6D11",
            "Sedang (200–400 kkal)": "#854F0B",
            "Tinggi (≥400 kkal)":    "#A32D2D"
        },
        title=f"Distribusi {len(df_foods)} makanan berdasarkan kategori kalori per porsi"
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(height=370)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Interpretasi Hasil**")
    top_kat = kat_count.sort_values("Jumlah", ascending=False).iloc[0]
    persen_top = round(top_kat["Jumlah"] / len(df_foods) * 100, 1)
    st.info(
       "Sebagian besar makanan dalam database termasuk kategori rendah kalori per porsi (<200 kkal). "
       "Hal ini bukan berarti makanan tersebut ringan secara nutrisi, melainkan karena banyak data "
       "dicatat dalam satuan per item kecil — seperti 1 tusuk sate, 1 buah klepon, atau 1 potong kue. "
       "Makanan kategori tinggi kalori (≥400 kkal) umumnya adalah hidangan nasi lengkap seperti "
       "Nasi Rames Bali dan Nasi Campur Bali yang mencakup nasi, lauk, dan pelengkap dalam satu porsi."
    )
    st.divider()

    # ── Perbandingan kandungan nutrisi ──
    st.markdown("#### Perbandingan Kandungan Nutrisi")
    st.markdown("""
    Makanan apa yang memberikan kandungan nutrisi terbesar dalam satu porsi standar,
    dan makanan apa yang relatif lebih rendah?

    Analisis ini dapat digunakan untuk mengidentifikasi makanan yang berpotensi menjadi
    sumber energi, protein, lemak, atau karbohidrat utama dalam pola konsumsi harian.
    """)

    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        rank_by = st.selectbox(
            "Urutkan berdasarkan:",
            ["calories", "protein", "fat", "carbohydrate"],
            format_func=lambda x: {
                "calories": "Kalori (kkal)", "protein": "Protein (g)",
                "fat": "Lemak (g)", "carbohydrate": "Karbohidrat (g)"
            }[x]
        )
    with cr2:
        kat_exp = st.multiselect(
            "Filter kategori kalori:",
            ["Rendah (<200 kkal)", "Sedang (200–400 kkal)", "Tinggi (≥400 kkal)"],
            default=["Rendah (<200 kkal)", "Sedang (200–400 kkal)", "Tinggi (≥400 kkal)"]
        )
    with cr3:
        daerah_exp = st.selectbox(
            "Filter asal daerah:",
            ["Semua"] + sorted(df_foods["asal_daerah"].dropna().unique().tolist()),
            key="daerah_tab2"
        )

    df_exp = df_foods[df_foods["kategori_kalori"].isin(kat_exp)].copy()
    if daerah_exp != "Semua":
        df_exp = df_exp[df_exp["asal_daerah"] == daerah_exp]
    df_exp = df_exp.sort_values(rank_by, ascending=False)

    unit  = {"calories": "kkal", "protein": "g", "fat": "g", "carbohydrate": "g"}[rank_by]
    label = {"calories": "Kalori", "protein": "Protein", "fat": "Lemak", "carbohydrate": "Karbohidrat"}[rank_by]

    # Tampilkan top 20 supaya grafik tidak terlalu panjang
    n_show = st.slider("Jumlah makanan ditampilkan:", 10, min(50, len(df_exp)), 20, 5)
    df_top = df_exp.head(n_show)

    if df_top.empty:
        st.info("Tidak ada data yang sesuai dengan filter yang dipilih.")
    else:
        fig2 = px.bar(
            df_top, x=rank_by, y="food_name", orientation="h",
            color="kategori_kalori",
            color_discrete_map={
                "Rendah (<200 kkal)":    "#3B6D11",
                "Sedang (200–400 kkal)": "#854F0B",
                "Tinggi (≥400 kkal)":    "#A32D2D"
            },
            text=rank_by,
            hover_data=["serving_description", "asal_daerah"],
            labels={
                "food_name":      "Makanan",
                rank_by:          f"{label} ({unit}) per porsi",
                "kategori_kalori":"Kategori Kalori",
                "serving_description": "Porsi",
                "asal_daerah":    "Daerah"
            },
            title=f"Top {n_show} makanan — {label.lower()} per porsi standar"
        )
        fig2.update_traces(texttemplate=f"%{{text:.1f}} {unit}", textposition="outside")
        fig2.update_layout(height=max(400, n_show * 22), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

        top1  = df_top.iloc[0]
        last1 = df_exp.iloc[-1]
        selisih = top1[rank_by] - last1[rank_by]
        st.markdown("**Interpretasi Hasil**")
        st.info(
            f"Makanan dengan {label.lower()} tertinggi adalah **{top1['food_name']}** "
            f"({top1[rank_by]:.1f} {unit}, {top1['serving_description']}), "
            f"sedangkan yang terendah adalah **{last1['food_name']}** "
            f"({last1[rank_by]:.1f} {unit}, {last1['serving_description']}). "
            f"Perlu dicatat bahwa perbedaan ini sebagian dipengaruhi oleh perbedaan ukuran porsi "
            f"antardata — makanan yang dicatat per porsi besar (misalnya 1 mangkok atau 1 piring) "
            f"cenderung memiliki nilai lebih tinggi dibanding yang dicatat per item kecil, "
            f"sehingga perbandingan ini perlu dibaca dengan mempertimbangkan deskripsi porsi masing-masing makanan."
        )
    st.divider()

    # ── Indeks Efisiensi Protein ──
    st.markdown("#### Indeks Efisiensi Protein")
    st.markdown("""
    Makanan apa yang memberikan kandungan protein paling efisien
    dibandingkan jumlah energi yang dikonsumsi?

    Indeks efisiensi protein digunakan untuk mengidentifikasi makanan yang mampu
    menyediakan protein dalam jumlah tinggi tanpa disertai peningkatan kalori yang berlebihan,
    sehingga relevan bagi individu yang ingin menjaga berat badan atau meningkatkan asupan protein.
    """)

    df_eff = df_foods.copy()
    df_eff = df_eff[df_eff["calories"] > 0]  # hindari pembagian nol
    df_eff = df_eff.sort_values("efisiensi_protein", ascending=False)

    n_eff = st.slider("Jumlah makanan ditampilkan:", 10, min(50, len(df_eff)), 20, 5, key="n_eff")
    df_eff_top = df_eff.head(n_eff)

    fig3 = px.bar(
        df_eff_top, x="efisiensi_protein", y="food_name", orientation="h",
        color="efisiensi_protein",
        color_continuous_scale=["#EF9F27", "#3B6D11"],
        text="efisiensi_protein",
        hover_data=["serving_description", "calories", "protein"],
        labels={
            "food_name":          "Makanan",
            "efisiensi_protein":  "Gram protein per 100 kkal",
            "serving_description":"Porsi",
            "calories":           "Kalori (kkal)",
            "protein":            "Protein (g)"
        },
        title=f"Top {n_eff} indeks efisiensi protein — gram protein per 100 kkal"
    )
    fig3.update_traces(texttemplate="%{text:.2f} g/100 kkal", textposition="outside")
    fig3.update_layout(height=max(400, n_eff * 22), coloraxis_showscale=False,
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    top_eff  = df_eff.iloc[0]
    last_eff = df_eff.iloc[-1]
    st.markdown("**Interpretasi Hasil**")
    st.info(
        f"**{top_eff['food_name']}** mencatat indeks tertinggi ({top_eff['efisiensi_protein']:.2f} g/100 kkal), "
        f"yang menunjukkan kandungan protein tinggi relatif terhadap kalorinya. "
        f"Makanan berbahan dasar telur, ayam, atau ikan umumnya mendominasi posisi teratas indeks ini "
        f"karena tinggi protein namun tidak mengandung banyak lemak atau karbohidrat tambahan. "
        f"Sebaliknya, makanan berbasis tepung dan gula seperti kue tradisional cenderung memiliki "
        f"indeks sangat rendah karena hampir seluruh kalorinya berasal dari karbohidrat, bukan protein."
    )
    st.divider()

    # ── Distribusi per daerah ──
    st.markdown("#### Distribusi Makanan per Daerah Asal")
    daerah_count = df_foods["asal_daerah"].value_counts().reset_index()
    daerah_count.columns = ["Daerah", "Jumlah"]
    fig_daerah = px.bar(
        daerah_count, x="Jumlah", y="Daerah", orientation="h",
        color="Jumlah",
        color_continuous_scale=["#EF9F27", "#3B6D11"],
        text="Jumlah",
        title="Jumlah makanan dalam database berdasarkan daerah asal"
    )
    fig_daerah.update_traces(textposition="outside")
    fig_daerah.update_layout(height=500, coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_daerah, use_container_width=True)

    with st.expander("Lihat seluruh tabel database makanan"):
        st.dataframe(
            df_foods[["food_name","serving_description","calories","protein","fat","carbohydrate","asal_daerah","kategori_kalori"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "food_name":           "Nama Makanan",
                "serving_description": "Porsi Standar",
                "calories":            st.column_config.NumberColumn("Kalori (kkal)", format="%.0f"),
                "protein":             st.column_config.NumberColumn("Protein (g)",   format="%.1f"),
                "fat":                 st.column_config.NumberColumn("Lemak (g)",     format="%.1f"),
                "carbohydrate":        st.column_config.NumberColumn("Karbo (g)",     format="%.1f"),
                "asal_daerah":         "Daerah",
                "kategori_kalori":     "Kategori",
            }
        )

# ════════════════════════════════════════
# TAB 3 — PERFORMA MODEL
# Tidak ada perubahan dari versi sebelumnya
# ════════════════════════════════════════
with tab3:
    st.subheader("Evaluasi Performa Model YOLOv8")
    st.markdown("""
    **Pertanyaan Bisnis**

    Apakah model YOLOv8 mampu mengenali berbagai jenis makanan Indonesia
    secara konsisten dan akurat sehingga dapat digunakan sebagai dasar
    dalam sistem estimasi nutrisi otomatis?

    Analisis ini mengevaluasi tingkat keberhasilan model dalam mendeteksi objek makanan
    serta mengidentifikasi kelas yang masih memerlukan perbaikan pada proses pelatihan.
    """)

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Total kelas",     "15")
    pm2.metric("Total gambar",    "4.632")
    pm3.metric("mAP50 rata-rata", "0.987")
    pm4.metric("Precision",       "0.985")
    pm5.metric("Recall",          "0.977")

    st.markdown("**Ringkasan Performa**")
    st.success(
        "Model YOLOv8 mencapai nilai rata-rata mAP50 sebesar 0.987 pada 15 kelas makanan Indonesia. "
        "Seluruh kelas memperoleh nilai mAP50 di atas 0.94, menunjukkan bahwa performa model relatif konsisten "
        "antar kelas dan tidak terdapat kelas dengan tingkat akurasi yang sangat rendah. "
        "Hasil ini mengindikasikan bahwa model memiliki kemampuan deteksi yang baik terhadap variasi objek makanan pada dataset."
    )
    st.divider()

    st.markdown("#### Visualisasi Hasil Pelatihan Model")
    st.markdown(
        "Visualisasi berikut dihasilkan selama proses pelatihan model YOLOv8 "
        "dan mencerminkan karakteristik performa model secara keseluruhan."
    )
    IMAGE_FILES = {
        "confusion_matrix_normalized.png": "Confusion Matrix — Sebaran prediksi per kelas",
        "results.png":                     "Grafik Pelatihan — Loss, Precision, Recall, mAP per epoch",
        "BoxPR_curve.png":                 "Precision-Recall Curve",
        "BoxF1_curve.png":                 "F1 Curve — Keseimbangan precision dan recall",
    }
    IMG_FOLDER = Path("images")
    if not IMG_FOLDER.exists():
        st.info(
            "Folder 'images/' belum ditemukan. Buat folder bernama 'images' "
            "di direktori yang sama dengan Dashboard.py dan tempatkan file hasil training."
        )
    else:
        found = {k: v for k, v in IMAGE_FILES.items() if (IMG_FOLDER / k).exists()}
        if not found:
            st.info("Folder 'images/' ditemukan, namun belum ada file gambar.")
        else:
            img_sel = st.selectbox("Pilih visualisasi:", list(found.values()))
            img_key = [k for k, v in found.items() if v == img_sel][0]
            st.image(str(IMG_FOLDER / img_key), caption=img_sel, use_container_width=True)
            insights = {
                "Confusion Matrix — Sebaran prediksi per kelas": (
                    "Diagonal utama yang dominan menunjukkan model memprediksi kelas yang benar "
                    "secara konsisten. Kesalahan klasifikasi terbanyak terjadi pada kelas bakso dan pempek "
                    "akibat kemiripan visual antar keduanya."
                ),
                "Grafik Pelatihan — Loss, Precision, Recall, mAP per epoch": (
                    "Loss yang menurun konsisten dan metrik yang meningkat menunjukkan model belajar "
                    "tanpa overfitting. Konvergensi tercapai sebelum epoch ke-50."
                ),
                "Precision-Recall Curve": (
                    "Kurva mendekati pojok kanan atas menunjukkan model mempertahankan precision tinggi "
                    "meski recall ditingkatkan. AUC yang besar mengkonfirmasi stabilitas model."
                ),
                "F1 Curve — Keseimbangan precision dan recall": (
                    "F1 score tinggi pada berbagai threshold menunjukkan keseimbangan baik antara "
                    "precision dan recall, penting untuk meminimalkan false positive dan false negative."
                ),
            }
            if img_sel in insights:
                st.markdown("**Interpretasi**")
                st.info(insights[img_sel])
    st.divider()

    st.markdown("#### Perbandingan Metrik Evaluasi per Kelas")
    pf1, pf2 = st.columns(2)
    with pf1:
        metrik_sel = st.selectbox("Pilih metrik evaluasi:", ["mAP50","mAP50_95","Precision","Recall"])
    with pf2:
        sort_dir = st.radio("Urutan tampilan:", ["Tertinggi","Terendah"], horizontal=True)

    df_m  = df_model.sort_values(metrik_sel, ascending=(sort_dir == "Terendah"))
    fig4  = px.bar(
        df_m, x=metrik_sel, y="Kelas", orientation="h",
        text=metrik_sel, color=metrik_sel,
        color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.92, 1.0],
        title=f"Nilai {metrik_sel} per kelas makanan"
    )
    fig4.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig4.update_layout(height=500, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

    best  = df_model.loc[df_model["mAP50"].idxmax()]
    worst = df_model.loc[df_model["mAP50"].idxmin()]
    selisih_map = best["mAP50"] - worst["mAP50"]
    st.markdown("**Interpretasi Hasil**")
    st.info(
        f"Kelas **{best['Kelas']}** mencapai mAP50 = {best['mAP50']:.3f}, "
        f"sementara **{worst['Kelas']}** memperoleh nilai terendah ({worst['mAP50']:.3f}). "
        f"Kedua kelas yang memiliki performa relatif lebih rendah — bakso dan pempek — "
        f"memiliki karakteristik visual yang cukup mirip satu sama lain: berbentuk bulat, "
        f"berwarna coklat keabu-abuan, dan sering muncul dalam konteks penyajian yang serupa. "
        f"Kemiripan ini kemungkinan menjadi faktor utama yang membuat model lebih sering keliru "
        f"pada kedua kelas tersebut dibanding kelas lainnya."
    )
    st.markdown("**Implikasi**")
    st.info(
        f"Kelas **{worst['Kelas']}** memiliki performa terendah dibandingkan kelas lainnya sehingga "
        f"dapat dijadikan fokus utama pada pengembangan model berikutnya. "
        f"Perbaikan dapat diarahkan pada peningkatan kualitas anotasi, penambahan variasi citra, "
        f"serta pengumpulan contoh yang merepresentasikan kondisi nyata yang lebih beragam."
    )
    st.divider()

    st.markdown("#### Hubungan Volume Data Pelatihan dan Akurasi")
    fig5 = px.scatter(
        df_model, x="Images", y="mAP50",
        text="Kelas", size="Images",
        color="mAP50",
        color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.94, 1.0],
        title="Jumlah gambar pelatihan vs nilai mAP50 per kelas",
        height=420
    )
    fig5.update_traces(textposition="top center")
    st.plotly_chart(fig5, use_container_width=True)

    corr = df_model["Images"].corr(df_model["mAP50"])
    kuat = "lemah" if abs(corr) < 0.3 else ("sedang" if abs(corr) < 0.6 else "kuat")
    st.markdown("**Interpretasi Hasil**")
    st.info(
        f"Nilai korelasi {corr:.3f} mengindikasikan hubungan yang {kuat} antara jumlah gambar "
        f"dan mAP50. Kelas **{worst['Kelas']}** justru memiliki gambar terbanyak "
        f"({int(worst['Images'])} gambar) namun mAP50 terendah, sementara beberapa kelas dengan "
        f"gambar lebih sedikit seperti nasi goreng dan soto mencapai mAP50 sempurna atau mendekati. "
        f"Pola ini menunjukkan bahwa setelah jumlah data melewati ambang batas tertentu, "
        f"faktor yang lebih menentukan adalah keragaman visual gambar — variasi sudut pengambilan, "
        f"pencahayaan, dan latar belakang — bukan sekadar penambahan jumlah."
    )
    st.divider()

    st.markdown("#### Analisis Precision-Recall per Kelas")
    fig6 = px.scatter(
        df_model, x="Precision", y="Recall",
        size="Images", color="mAP50", text="Kelas",
        color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.94, 1.0],
        title="Sebaran precision dan recall per kelas (ukuran titik: jumlah gambar)",
        height=450
    )
    fig6.update_traces(textposition="top center")
    fig6.add_hline(y=0.95, line_dash="dot", line_color="gray",
                   annotation_text="Recall = 0.95", annotation_position="bottom right")
    fig6.add_vline(x=0.95, line_dash="dot", line_color="gray",
                   annotation_text="Precision = 0.95")
    st.plotly_chart(fig6, use_container_width=True)

    below_recall = df_model[df_model["Recall"]    < 0.95]["Kelas"].tolist()
    below_prec   = df_model[df_model["Precision"] < 0.95]["Kelas"].tolist()
    temuan_pr = "Mayoritas kelas berada pada zona ideal (precision dan recall > 0.95). "
    if below_recall: temuan_pr += f"Di bawah threshold recall: **{', '.join(below_recall)}**. "
    if below_prec:   temuan_pr += f"Di bawah threshold precision: **{', '.join(below_prec)}**."
    jumlah_ideal = len(
        df_model[
            (df_model["Precision"] >= 0.95) &
            (df_model["Recall"] >= 0.95)
       ]
    )

    st.markdown("**Interpretasi Hasil**")
    st.info(
        f"Sebanyak **{jumlah_ideal} dari {len(df_model)} kelas** memiliki precision dan recall di atas 0.95. "
        f"Kelas bakso menjadi satu-satunya yang berada di luar zona ideal pada sisi recall (0.833), "
        f"artinya model masih melewatkan sebagian objek bakso yang seharusnya terdeteksi. "
        f"Ini berbeda dengan precision yang tetap tinggi (0.957), yang berarti ketika model "
        f"mendeteksi bakso, prediksinya biasanya benar — masalahnya ada pada sensitivitas, bukan ketelitian."
    )
    st.markdown("**Implikasi**")
    st.info(
        "Profil recall rendah pada kelas bakso mengarah pada prioritas yang spesifik: "
        "bukan memperbanyak data secara keseluruhan, melainkan menambah variasi gambar bakso "
        "yang lebih sulit dikenali — misalnya bakso dalam kuah gelap, bakso potong, "
        "atau bakso yang sebagian tertutup bahan lain."
    )

    with st.expander("Lihat tabel lengkap metrik evaluasi per kelas"):
        st.dataframe(df_model, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# KESIMPULAN AKHIR
# ════════════════════════════════════════
st.divider()
st.markdown("### Kesimpulan")
st.success(f"""
Berdasarkan seluruh analisis yang disajikan dalam dashboard ini, diperoleh kesimpulan utama sebagai berikut.

Pertama, integrasi database {len(df_foods)} makanan Indonesia dan AKG Permenkes No. 28 Tahun 2019
memungkinkan estimasi kontribusi nutrisi secara personal dan terukur untuk berbagai jenis makanan.

Kedua, kalkulator nutrisi kini mendukung seluruh makanan dalam database — termasuk makanan
yang disajikan per item seperti sate tusuk dan klepon — dengan logika porsi yang adaptif.

Ketiga, model YOLOv8 mencapai rata-rata mAP50 sebesar 0.987 pada 15 kelas makanan Indonesia,
menunjukkan performa deteksi yang tinggi dan konsisten.

Keempat, keragaman visual data pelatihan terbukti lebih berpengaruh terhadap akurasi
dibandingkan volume data semata.

Kelima, sistem NutriCitra menerjemahkan hasil deteksi menjadi informasi gizi yang kontekstual
berdasarkan profil AKG pengguna — membedakannya dari aplikasi deteksi makanan pada umumnya.
""")

# ════════════════════════════════════════
# KETERANGAN DATA
# ════════════════════════════════════════
st.divider()
with st.expander("Keterangan Sumber Data dan Metodologi"):
    st.markdown("""
    **Database Makanan Indonesia**
    Database berisi 289 makanan Indonesia. Data nutrisi disajikan per porsi standar
    masing-masing makanan (misalnya "1 Porsi (270 g)" atau "1 Tusuk").
    Karena data bersumber dari pengumpulan manual dan web scraping, nilai nutrisi
    merupakan estimasi dan dapat bervariasi tergantung resep serta metode pengolahan.

    **Angka Kecukupan Gizi (AKG)**
    Nilai AKG bersumber dari Peraturan Menteri Kesehatan RI Nomor 28 Tahun 2019.
    Disesuaikan berdasarkan kelompok usia, jenis kelamin, dan kondisi fisiologis pengguna.

    **Model YOLOv8**
    Dilatih pada dataset 15 kelas makanan Indonesia (4.632 gambar).
    Metrik evaluasi (Precision, Recall, mAP50, mAP50-95) dihitung pada data validasi.
    """)

st.caption(
    f"Database: {len(df_foods)} makanan Indonesia | "
    "AKG: Permenkes No. 28 Tahun 2019 | "
    "Model: YOLOv8 — 4.632 gambar — 15 kelas"
)
