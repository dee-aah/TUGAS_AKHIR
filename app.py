import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from streamlit_autorefresh import st_autorefresh
from google.auth.credentials import AnonymousCredentials

# CONFIG

MODEL_PATH = "models/model_logreg.pkl"
TFIDF_PATH = "models/tfidf.pkl"
REFRESH_INTERVAL = 5000
label_map = {0: "Ham", 1: "Spam", 2: "Toxic"}

st.set_page_config(page_title="YouTube Spam & Toxic Detector", layout="wide")

# LOAD MODEL

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

model, tfidf = load_resources()

# SESSION STATE

def init_state():
    states = {
        "is_running": False,
        "next_page_token": None,
        "start_time": None,
        "all_comments": pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"]),
        "manual_history": pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"])
    }
    for k, v in states.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# CLASSIFICATION

def classify_comment(text: str) -> str:
    url_pattern = r"(https?://\S+|www\.\S+|\S+\.(com|net|org|id|me|info))"
    if re.search(url_pattern, text, re.IGNORECASE):
        return "Spam"

    if model and tfidf:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        return label_map[pred]

    return "N/A"


# YOUTUBE API

def get_live_chat_id(api_key, video_id):
    try:
        yt = build("youtube", "v3", developerKey=api_key, credentials=AnonymousCredentials())

        res = yt.videos().list(
            part="liveStreamingDetails,snippet",
            id=video_id
        ).execute()

        if not res.get("items"):
            return None, None

        item = res["items"][0]
        live_chat_id = item.get("liveStreamingDetails", {}).get("activeLiveChatId")
        title = item.get("snippet", {}).get("title")

        return live_chat_id, title

    except Exception as e:
        st.error(f"Gagal mengambil live chat ID: {e}")
        return None, None
    
from googleapiclient.discovery import build
from google.auth.credentials import AnonymousCredentials

def get_live_chat_messages(api_key, live_chat_id, page_token=None):
    try:
        yt = build(
            "youtube",
            "v3",
            developerKey=api_key,
            credentials=AnonymousCredentials()
        )

        response = yt.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet,authorDetails",
            pageToken=page_token
        ).execute()

        messages = []

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            msg = snippet.get("displayMessage")

            if msg:
                messages.append(msg)

        next_token = response.get("nextPageToken")
        return messages, next_token

    except Exception as e:
        st.error(f"Gagal mengambil pesan live chat: {e}")
        return [], None




# UI STYLE

st.markdown("""
<style>
.chat-bubble {
    display: inline-block;
    padding:10px 14px;
    border-radius:10px;
    margin-bottom:6px;
    max-width: 80%; 
}
.chat-ham { background:#08c800; color:black; }
.chat-spam { background:#FFD41D; color:black; }
.chat-toxic { background:#ff4d4d; color:black; }
.chat-meta { font-size:0.7rem; color:#666; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>YouTube Live <a style='color:#FFD41D;'>Spam</a> & <a style='color:#ff4d4d;'>Toxic</a> Detector</h1>",
    unsafe_allow_html=True
    )
st.markdown("""
Aplikasi ini dirancang untuk memoderasi percakapan dengan mendeteksi kata-kata secara real-time pada Live Chat YouTube maupun input manual. Sistem akan secara otomatis mengklasifikasikan pesan ke dalam kategori: 
- Ham: Pesan yang aman dan relevan.
- Spam: Pesan yang mengandung unsur promosi atau gangguan.
- Toxic: Pesan yang mengandung kata-kata kasar atau tidak pantas.
""")


# TABS

tab1, tab2 = st.tabs([" Live Monitor", " Deteksi Manual"])

# TAB 1 - LIVE MONITOR

with tab1:
    with st.expander("⚙️ Pengaturan Live"):
        # api_key = st.secrets.get("YOUTUBE_API_KEY", "")
        api_key = st.text_input("YouTube API Key", type="password")
        video_id = st.text_input("Masukkan YouTube Video ID" ,placeholder="Contoh: dQw4w9WgXcQ")

        # c1, c2 = st.columns(2)
        # if c1.button(":green[▶] Mulai"):
        #     st.session_state.is_running = True
        #     st.session_state.start_time = datetime.now()
        #     st.session_state.all_comments = pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"])
        #     st.rerun()

        # if c2.button(":red[⏹] Berhenti"):
        #     st.session_state.is_running = False
        #     st.rerun()
        c1, c2, = st.columns([2, 2,])

        with c1:
            if st.button(":green[ ▶ ] Mulai", use_container_width=True):
                st.session_state.is_running = True
                st.session_state.start_time = datetime.now()
                st.session_state.all_comments = pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"])
                st.session_state.next_page_token = None
                st.rerun()
                pass

        with c2:
            if st.button(":red[⏹] Berhenti", use_container_width=True):
                st.session_state.is_running = False
                st.rerun()
                pass

    if st.session_state.is_running:
        elapsed = datetime.now() - st.session_state.start_time
        remaining = timedelta(minutes=20) - elapsed

        st.success(f"⏱ Durasi: {str(elapsed).split('.')[0]} / 20:00")
        elapsed = datetime.now() - st.session_state.start_time
        if elapsed > timedelta(minutes=20):
            st.warning("⏱ Waktu monitoring selesai")
            st.session_state.is_running = False
        
        else:
            chat_id, title = get_live_chat_id(api_key, video_id)
            if chat_id:
                st.success(f"📺 {title}")
                messages, next_token = get_live_chat_messages(
                    api_key,
                    chat_id,
                    st.session_state.next_page_token
                )
            st.session_state.next_page_token = next_token

            for msg in messages:
                label = classify_comment(msg)
                st.session_state.all_comments.loc[len(st.session_state.all_comments)] = [
                datetime.now(), msg, label
                ]
            st_autorefresh(interval=REFRESH_INTERVAL, key="refresh")

    df = st.session_state.all_comments
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", len(df))
    m2.metric("🟢 Ham", (df["Prediksi"] == "Ham").sum())
    m3.metric("🟡 Spam", (df["Prediksi"] == "Spam").sum())
    m4.metric("🔴 Toxic", (df["Prediksi"] == "Toxic").sum())


    # Tampilkan chat terakhir
    for _, row in st.session_state.all_comments.tail(10).iloc[::-1].iterrows():
        style = "chat-ham" if row["Prediksi"] == "Ham" else \
                "chat-spam" if row["Prediksi"] == "Spam" else "chat-toxic"
        st.markdown(
            f'<div class="chat-meta">{row["Waktu"].strftime("%H:%M")} | {row["Prediksi"]}</div>'
            f'<div class="chat-bubble {style}">{row["Komentar"]}</div>',
            unsafe_allow_html=True
        )


# TAB 2 - MANUAL

with tab2:
    input_user = st.text_area("Masukkan teks")

    if st.button("Analisis", type="primary"):
        label_hasil = classify_comment(input_user)

        st.session_state.manual_history.loc[len(st.session_state.manual_history)] = [
            datetime.now(), input_user, label_hasil
        ]

        if label_hasil == "Ham":
            st.success(f"✅ Hasil: {label_hasil}")
            style = "chat-ham"
        elif label_hasil == "Spam":
            st.warning(f"🟡 Hasil: {label_hasil}")
            style = "chat-spam"
        else:
            st.error(f"🔴 Hasil: {label_hasil}")
            style = "chat-toxic"

        st.markdown(
            f'<div class="chat-bubble {style}">{input_user}</div>',
            unsafe_allow_html=True
        )

# SUMMARY FUNCTION

def show_summary(df, title):
    if df.empty:
        return

    st.divider()
    st.subheader(f"📊 Kesimpulan Analisis {title}")

    total = len(df)
    ham = (df["Prediksi"] == "Ham").sum()
    spam = (df["Prediksi"] == "Spam").sum()
    toxic = (df["Prediksi"] == "Toxic").sum()

    p_spam = spam / total * 100
    p_toxic = toxic / total * 100

    c1, c2 = st.columns([1, 2])

    with c1:
        st.write("**Statistik:**")
        st.write(f"- 🟢 Ham: {ham}")
        st.write(f"- 🟡 Spam: {spam}")
        st.write(f"- 🔴 Toxic: {toxic}")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"{title}.csv",
            "text/csv"
        )

    with c2:
        chart = pd.DataFrame(
            {"Jumlah": [ham, spam, toxic]},
            index=["Ham", "Spam", "Toxic"]
        )
        st.bar_chart(chart)

    if title == "Live Chat":
        if p_toxic > 20 or (p_spam + p_toxic) > 40:
            st.error(
            "🚨 **KONDISI BAHAYA**\n\n"
            "Mayoritas chat mengandung pesan **toxic dan/atau spam**.\n\n"
            "**Rekomendasi:**\n"
            "- Aktifkan *Slow Mode*\n"
            "- Jauhkan dari anak - anak dibawah umur\n"
            "- Moderasi manual atau otomatis\n"
            "- Blokir akun yang berulang kali melanggar\n"
            "- Nonaktifkan link di live chat sementara"
            )

        elif p_toxic > 5 or p_spam > 15:
            st.warning(
            "⚠ **KONDISI WASPADA**\n\n"
            "Terdapat peningkatan pesan **spam atau toxic**, "
            "namun masih dalam batas yang dapat dikendalikan.\n\n"
            "**Rekomendasi:**\n"
            "- Pantau live chat secara berkala\n"
            "- Bukan Tontonan anak - anak dibawah umur\n"
            "- Siapkan moderator\n"
            "- Batasi pengiriman pesan beruntun"
            )

        else:
            st.success(
            "✅ **KONDISI AMAN**\n\n"
            "Interaksi chat didominasi pesan **positif dan relevan**.\n\n"
            "**Rekomendasi:**\n"
            "- Live chat dapat berjalan normal\n"
            "- Tetap lakukan pemantauan rutin"
            )

# SHOW SUMMARY

with tab1:
    if not st.session_state.is_running:
        show_summary(st.session_state.all_comments, "Live Chat")

with tab2:
    show_summary(st.session_state.manual_history, "Deteksi Manual")
