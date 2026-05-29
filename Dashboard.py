import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        elif kondisi == "Hamil trimester 1":  row = row[row["hamil_1_13"]==1]
        elif kondisi == "Hamil trimester 2":  row = row[row["hamil_14_27"]==1]
        elif kondisi == "Hamil trimester 3":  row = row[row["hamil_28_41"]==1]
        elif kondisi == "Menyusui 0-6 bulan": row = row[row["menyusui_6bl_pertama"]==1]
        elif kondisi == "Menyusui 7-12 bulan":row = row[row["menyusui_6bl_kedua"]==1]
    if row.empty: return None
    r = row.iloc[0]
    return {"kal":int(r["energi_kkal"]), "protein":int(r["protein_g"]),
            "lemak":float(r["lemak_total_g"]), "karbo":int(r["karbohidrat_g"])}

# ─────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────
st.set_page_config(page_title="NutriCitra", page_icon="🍛", layout="wide")
st.title("🍛 NutriCitra — Deteksi Makanan & Estimasi Kalori")
st.caption("YOLOv8 · TKPI 2017 · AKG Permenkes 2019")

df_akg = load_akg()

# SIDEBAR
with st.sidebar:
    st.header("👤 Profil Pengguna")
    gender    = st.radio("Jenis kelamin:", ["Laki-laki","Perempuan"])
    gender_int= 0 if gender == "Laki-laki" else 1
    usia      = st.slider("Usia (tahun):", 1, 80, 25)
    usia_str  = f"{usia} tahun"
    kondisi   = "Normal"
    if gender == "Perempuan":
        kondisi = st.selectbox("Kondisi:", ["Normal","Hamil trimester 1",
                               "Hamil trimester 2","Hamil trimester 3",
                               "Menyusui 0-6 bulan","Menyusui 7-12 bulan"])
    akg = get_akg(df_akg, usia_str, gender_int, kondisi)
    if akg:
        st.divider()
        st.subheader("📊 AKG Harianmu")
        c1,c2 = st.columns(2)
        c1.metric("Kalori",      f"{akg['kal']} kkal")
        c1.metric("Lemak",       f"{akg['lemak']} g")
        c2.metric("Protein",     f"{akg['protein']} g")
        c2.metric("Karbohidrat", f"{akg['karbo']} g")
    else:
        st.warning("Data AKG tidak ditemukan.")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🍽️ Kalkulator Nutrisi",
    "🔍 Eksplorasi Makanan",
    "📈 Performa Model",
    "📋 Data Dictionary",
])

# ════════════════════════════════════════
# TAB 1 — KALKULATOR NUTRISI
# ════════════════════════════════════════
with tab1:
    st.subheader("Filter & Pilih Makanan")
    cf1, cf2 = st.columns(2)
    with cf1:
        kat_filter = st.radio("Kategori kalori (per 100g):",
            ["Semua","Rendah (<100 kkal)","Sedang (100–199 kkal)","Tinggi (≥200 kkal)"],
            horizontal=True)
    with cf2:
        sumber_filter = st.radio("Sumber data:",
            ["Semua","TKPI_2017","TKPI_komposit"], horizontal=True)

    df_f = df_foods.copy()
    if kat_filter == "Rendah (<100 kkal)":    df_f = df_f[df_f["kal"] < 100]
    elif kat_filter == "Sedang (100–199 kkal)":df_f = df_f[(df_f["kal"]>=100)&(df_f["kal"]<200)]
    elif kat_filter == "Tinggi (≥200 kkal)":  df_f = df_f[df_f["kal"] >= 200]
    if sumber_filter != "Semua":              df_f = df_f[df_f["sumber"]==sumber_filter]

    if df_f.empty:
        st.info("Tidak ada makanan di kategori ini.")
    else:
        makanan_sel = st.selectbox("Pilih makanan:", df_f["name"].tolist())
        berat = st.slider("Estimasi berat porsi (gram):", 50, 500, 200, 10)
        food  = df_f[df_f["name"]==makanan_sel].iloc[0]
        fk    = berat / 100

        kal_p  = round(food["kal"]     * fk)
        prot_p = round(food["protein"] * fk, 1)
        lemak_p= round(food["lemak"]   * fk, 1)
        karbo_p= round(food["karbo"]   * fk, 1)

        st.divider()
        st.subheader(f"Kandungan nutrisi — {berat}g {makanan_sel}")

        if akg:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("🔥 Kalori",      f"{kal_p} kkal",  f"{round(kal_p/akg['kal']*100,1)}% AKG")
            m2.metric("🥩 Protein",     f"{prot_p} g",    f"{round(prot_p/akg['protein']*100,1)}% AKG")
            m3.metric("🧈 Lemak",       f"{lemak_p} g",   f"{round(lemak_p/akg['lemak']*100,1)}% AKG")
            m4.metric("🍚 Karbohidrat", f"{karbo_p} g",   f"{round(karbo_p/akg['karbo']*100,1)}% AKG")

            df_chart = pd.DataFrame({
                "Nutrisi": ["Kalori (kkal)","Protein (g)","Lemak (g)","Karbohidrat (g)"],
                "% AKG":   [round(v/t*100,1) for v,t in zip(
                    [kal_p,prot_p,lemak_p,karbo_p],
                    [akg["kal"],akg["protein"],akg["lemak"],akg["karbo"]])]
            })
            fig = px.bar(df_chart, x="Nutrisi", y="% AKG",
                color="% AKG", color_continuous_scale=["#3B6D11","#EF9F27","#A32D2D"],
                range_color=[0,50], text="% AKG",
                title=f"Pemenuhan AKG — {gender}, {usia} tahun, {kondisi}")
            fig.add_hline(y=25, line_dash="dash", line_color="orange",
                          annotation_text="25% (batas wajar 1 porsi)")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Lengkapi profil di sidebar untuk melihat % AKG.")
        st.caption(f"📚 Sumber: {'TKPI 2017' if food['sumber']=='TKPI_2017' else 'TKPI 2017 (estimasi komposit)'}")

# ════════════════════════════════════════
# TAB 2 — EKSPLORASI MAKANAN
# ════════════════════════════════════════
with tab2:
    st.subheader("Ranking & Perbandingan Makanan (per 100g)")
    cr1,cr2 = st.columns(2)
    with cr1:
        rank_by = st.selectbox("Ranking berdasarkan:",
            ["kal","protein","lemak","karbo"],
            format_func=lambda x:{"kal":"Kalori","protein":"Protein",
                                   "lemak":"Lemak","karbo":"Karbohidrat"}[x])
    with cr2:
        kat_exp = st.multiselect("Filter kategori:",
            ["Rendah","Sedang","Tinggi"], default=["Rendah","Sedang","Tinggi"])

    df_exp = df_foods[df_foods["kategori_kalori"].isin(kat_exp)].sort_values(rank_by, ascending=False)
    unit   = {"kal":"kkal","protein":"g","lemak":"g","karbo":"g"}[rank_by]
    label  = {"kal":"Kalori","protein":"Protein","lemak":"Lemak","karbo":"Karbohidrat"}[rank_by]

    fig2 = px.bar(df_exp, x=rank_by, y="name", orientation="h",
        color="kategori_kalori",
        color_discrete_map={"Rendah":"#3B6D11","Sedang":"#854F0B","Tinggi":"#A32D2D"},
        text=rank_by,
        labels={"name":"Makanan", rank_by:f"{label} ({unit})", "kategori_kalori":"Kategori"},
        title=f"Ranking {label} per 100g")
    fig2.update_traces(texttemplate=f"%{{text}} {unit}", textposition="outside")
    fig2.update_layout(height=480, yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════
# TAB 3 — PERFORMA MODEL
# ════════════════════════════════════════
with tab3:
    st.subheader("Performa Model YOLOv8")

    # Metrik ringkasan
    pm1,pm2,pm3,pm4,pm5 = st.columns(5)
    pm1.metric("Total kelas",    "15")
    pm2.metric("Total gambar",   "853")
    pm3.metric("mAP50 rata-rata","0.987")
    pm4.metric("Precision",      "0.985")
    pm5.metric("Recall",         "0.977")

    st.divider()

    # Filter & sort
    pf1,pf2 = st.columns(2)
    with pf1:
        metrik_sel = st.selectbox("Tampilkan metrik:",
            ["mAP50","mAP50_95","Precision","Recall"])
    with pf2:
        sort_dir = st.radio("Urutan:", ["Tertinggi dulu","Terendah dulu"], horizontal=True)

    df_m = df_model.sort_values(metrik_sel, ascending=(sort_dir=="Terendah dulu"))

    # Bar chart metrik per kelas
    color_map = df_m[metrik_sel].apply(
        lambda v: "#3B6D11" if v >= 0.99 else ("#185FA5" if v >= 0.97 else ("#854F0B" if v >= 0.95 else "#A32D2D"))
    )
    fig3 = px.bar(df_m, x=metrik_sel, y="Kelas", orientation="h",
        text=metrik_sel,
        color=metrik_sel,
        color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.92, 1.0],
        title=f"{metrik_sel} per kelas makanan")
    fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig3.update_layout(height=500, coloraxis_showscale=False,
                       yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Scatter: Precision vs Recall
    st.subheader("Precision vs Recall per kelas")
    st.caption("Ukuran titik = jumlah gambar training. Ideal: titik di pojok kanan atas.")
    fig4 = px.scatter(df_model, x="Precision", y="Recall",
        size="Images", color="mAP50", text="Kelas",
        color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.94, 1.0],
        title="Precision vs Recall (warna = mAP50, ukuran = jumlah gambar)",
        height=450)
    fig4.update_traces(textposition="top center")
    fig4.add_hline(y=0.95, line_dash="dot", line_color="gray", annotation_text="Recall 0.95")
    fig4.add_vline(x=0.95, line_dash="dot", line_color="gray", annotation_text="Precision 0.95")
    fig4.update_layout(coloraxis_colorbar_title="mAP50")
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # Jumlah gambar vs mAP50 — pertanyaan bisnis 3
    st.subheader("Apakah jumlah gambar mempengaruhi akurasi?")
    st.caption("Pertanyaan bisnis 3: korelasi jumlah gambar training vs mAP50 per kelas.")
    fig5 = px.scatter(df_model, x="Images", y="mAP50",
        text="Kelas", size="Images",
        color="mAP50", color_continuous_scale=["#A32D2D","#EF9F27","#3B6D11"],
        range_color=[0.94, 1.0],
        title="Jumlah gambar vs mAP50 per kelas", height=400)
    fig5.update_traces(textposition="top center")
    fig5.update_layout(coloraxis_colorbar_title="mAP50")
    st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # Tabel lengkap
    with st.expander("📋 Lihat tabel lengkap performa model"):
        st.dataframe(df_model, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 4 — DATA DICTIONARY
# ════════════════════════════════════════
with tab4:
    st.subheader("Data Dictionary — 15 Kelas Makanan NutriCitra")
    st.caption("Sumber: TKPI 2017 (Kemenkes RI) | TKPI_komposit = estimasi dari komponen TKPI")

    df_dict = df_foods[["name","kal","protein","lemak","karbo","kategori_kalori","sumber"]].copy()
    df_dict.columns = ["Nama Makanan","Kalori (kkal/100g)","Protein (g/100g)",
                       "Lemak (g/100g)","Karbo (g/100g)","Kategori Kalori","Sumber Data"]
    st.dataframe(df_dict, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Keterangan Sumber Data")
    st.markdown("""
- **TKPI_2017** — data diambil langsung dari Tabel Komposisi Pangan Indonesia 2017, Kemenkes RI
- **TKPI_komposit** — estimasi dari komponen penyusun makanan berdasarkan TKPI 2017 (untuk hidangan gabungan yang tidak punya entri tunggal)
    """)

    st.divider()
    st.subheader("AKG berdasarkan profil pengguna saat ini")
    if akg:
        df_akg_chart = pd.DataFrame({
            "Nutrisi": ["Kalori (kkal)","Protein (g)","Lemak (g)","Karbohidrat (g)"],
            "Target":  [akg["kal"], akg["protein"], akg["lemak"], akg["karbo"]]
        })
        fig6 = px.bar(df_akg_chart, x="Nutrisi", y="Target", color="Nutrisi",
            text="Target", title=f"Target AKG harian — {gender}, {usia} tahun, {kondisi}")
        fig6.update_traces(textposition="outside")
        fig6.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Lengkapi profil di sidebar untuk melihat AKG.")

st.divider()
st.caption("📚 TKPI 2017 (Kemenkes RI) | AKG: Permenkes No.28 Tahun 2019 | Model: YOLOv8 · 853 gambar · 15 kelas")