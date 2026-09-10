import streamlit as st
import pandas as pd
import re
import torch

from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
from google.auth.credentials import AnonymousCredentials
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="YouTube Spam & Toxic Detector",
    layout="wide"
)

MODEL_PATH = "models"
MAX_LENGTH = 32
REFRESH_INTERVAL = 5000

LABEL_MAP = {
    0: "Normal",
    1: "Spam",
    2: "Toxic"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_model():

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        model.to(device)
        model.eval()

        return tokenizer, model

    except Exception as e:

        st.error(
            f"Gagal memuat model IndoBERT: {e}"
        )

        return None, None


tokenizer, model = load_model()

def init_state():

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    if "next_page_token" not in st.session_state:
        st.session_state.next_page_token = None

    if "start_time" not in st.session_state:
        st.session_state.start_time = None

    if "all_comments" not in st.session_state:
        st.session_state.all_comments = pd.DataFrame(
            columns=[
                "Waktu",
                "Komentar",
                "Prediksi",
                "Confidence"
            ]
        )

    if "manual_history" not in st.session_state:
        st.session_state.manual_history = pd.DataFrame(
            columns=[
                "Waktu",
                "Komentar",
                "Prediksi",
                "Confidence"
            ]
        )


init_state()

leet_replace = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b"
}
def normalisasi_kalimat(kalimat):

    if pd.isna(kalimat):
        return ""

    text = str(kalimat).lower()

    hasil = []

    for kata in text.split():

        # Normalisasi angka yang menyerupai huruf
        if re.search(r"[a-zA-Z]", kata) and re.search(r"\d", kata):
            for angka, huruf in leet_replace.items():
                kata = kata.replace(angka, huruf)

        # Mengurangi huruf berulang
        kata = re.sub(r"(.)\1{2,}", r"\1", kata)


        hasil.append(kata)

    return " ".join(hasil)

# PREDIKSI

def classify_comment(text):

    text = str(text).strip()

    if not text:

        return "N/A", 0.0

    if tokenizer is None or model is None:

        return "N/A", 0.0

    # preprocessing
    text_clean = normalisasi_kalimat(
        text
    )

    try:

        # tokenisasi
        inputs = tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        # pindahkan tensor ke device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # prediksi
        with torch.no_grad():

            outputs = model(
                **inputs
            )

        # probabilitas
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        # kelas
        prediction = torch.argmax(
            probabilities
        ).item()

        # confidence
        confidence = probabilities[
            prediction
        ].item()

        label = LABEL_MAP.get(
            prediction,
            "N/A"
        )

        return label, confidence

    except Exception as e:

        st.error(
            f"Error prediksi: {e}"
        )

        return "N/A", 0.0


# YOUTUBE


def get_live_chat_id(
    api_key,
    video_id
):

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=api_key,
            credentials=AnonymousCredentials()
        )

        response = youtube.videos().list(
            part="liveStreamingDetails,snippet",
            id=video_id
        ).execute()

        if not response.get("items"):

            return None, None

        item = response["items"][0]

        live_chat_id = (
            item
            .get("liveStreamingDetails", {})
            .get("activeLiveChatId")
        )

        title = (
            item
            .get("snippet", {})
            .get("title", "YouTube Live")
        )

        return live_chat_id, title

    except Exception as e:

        st.error(
            f"Gagal mengambil Live Chat ID: {e}"
        )

        return None, None


def get_live_chat_messages(
    api_key,
    live_chat_id,
    page_token=None
):

    try:

        youtube = build(
            "youtube",
            "v3",
            developerKey=api_key,
            credentials=AnonymousCredentials()
        )

        params = {
            "liveChatId": live_chat_id,
            "part": "snippet,authorDetails"
        }

        if page_token:

            params["pageToken"] = page_token

        response = youtube.liveChatMessages().list(
            **params
        ).execute()

        messages = []

        for item in response.get(
            "items",
            []
        ):

            snippet = item.get(
                "snippet",
                {}
            )

            message = snippet.get(
                "displayMessage"
            )

            if message:

                messages.append(
                    message
                )

        next_token = response.get(
            "nextPageToken"
        )

        return messages, next_token

    except Exception as e:

        st.error(
            f"Gagal mengambil pesan Live Chat: {e}"
        )

        return [], None


# EXTRACT YOUTUBE VIDEO ID

def extract_video_id(url):

    if not url:

        return None

    patterns = [

        r"(?:youtube\.com/watch\?v=)([^&]+)",

        r"(?:youtu\.be/)([^?&]+)",

        r"(?:youtube\.com/live/)([^?&]+)",

        r"(?:youtube\.com/embed/)([^?&]+)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:

            return match.group(1)

    return None

# CSS

st.markdown(
    """
    <style>

    .chat-bubble {
        display: inline-block;
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 6px;
        max-width: 80%;
    }

    .chat-normal {
        background: #08c800;
        color: black;
    }

    .chat-spam {
        background: #FFD41D;
        color: black;
    }

    .chat-toxic {
        background: #ff4d4d;
        color: black;
    }

    .chat-meta {
        font-size: 0.75rem;
        color: #666;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# HEADER


st.markdown(
    """
    <h1 style="text-align:center;">
        YouTube Live
        <span style="color:#FFD41D;">
            Spam
        </span>
        &
        <span style="color:#ff4d4d;">
            Toxic
        </span>
        Detector
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Aplikasi ini dirancang untuk memoderasi percakapan dengan mendeteksi kata-kata secara real-time pada Live Chat YouTube maupun input manual. Sistem akan secara otomatis mengklasifikasikan pesan ke dalam kategori: 
- Normal: Pesan yang aman dan relevan.
- Spam: Pesan yang mengandung unsur promosi atau gangguan.
- Toxic: Pesan yang mengandung kata-kata kasar atau tidak pantas.
    """
)

# TABS

tab1, tab2, tab3 = st.tabs(
    [
        "🎥 Live Monitor",
        "🔍 Deteksi Manual",
        "📁 Upload File"
    ]
)

# TAB 1 - LIVE MONITOR

with tab1:

    st.subheader(
        "🎥 YouTube Live Monitor"
    )

    youtube_url = st.text_input(
        "Masukkan Link YouTube",
        placeholder=(
            "https://www.youtube.com/watch?v=..."
        )
    )

    # API KEY
    try:

        api_key = st.secrets[
            "YOUTUBE_API_KEY"
        ]

    except Exception:

        api_key = None

        st.warning(
            "YOUTUBE_API_KEY belum tersedia "
            "di Streamlit Secrets."
        )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "▶ Mulai",
            use_container_width=True
        ):

            if not youtube_url:

                st.warning(
                    "Masukkan link YouTube terlebih dahulu."
                )

            elif not api_key:

                st.error(
                    "API Key YouTube tidak ditemukan."
                )

            else:

                st.session_state.is_running = True

                st.session_state.start_time = (
                    datetime.now()
                )

                st.session_state.next_page_token = None

                st.session_state.all_comments = (
                    pd.DataFrame(
                        columns=[
                            "Waktu",
                            "Komentar",
                            "Prediksi",
                            "Confidence"
                        ]
                    )
                )

                st.rerun()


    with col2:

        if st.button(
            "⏹ Berhenti",
            use_container_width=True
        ):

            st.session_state.is_running = False

            st.rerun()

    # MONITORING

    if st.session_state.is_running:

        elapsed = (
            datetime.now()
            - st.session_state.start_time
        )

        max_duration = timedelta(
            minutes=20
        )

        if elapsed >= max_duration:

            st.session_state.is_running = False

            st.warning(
                "⏱ Monitoring 20 menit telah selesai."
            )

            st.rerun()


        remaining = (
            max_duration
            - elapsed
        )

        st.info(
            f"⏱ Durasi: "
            f"{str(elapsed).split('.')[0]}"
            f" / 20:00"
        )

        video_id = extract_video_id(
            youtube_url
        )

        if not video_id:

            st.error(
                "Link YouTube tidak valid."
            )

        else:

            chat_id, title = get_live_chat_id(
                api_key,
                video_id
            )

            if not chat_id:

                st.error(
                    "Live Chat tidak tersedia. "
                    "Pastikan video sedang live."
                )

            else:

                st.success(
                    f"🎥 {title}"
                )

                messages, next_token = (
                    get_live_chat_messages(
                        api_key,
                        chat_id,
                        st.session_state.next_page_token
                    )
                )

                st.session_state.next_page_token = (
                    next_token
                )

                # CEGAH DUPLIKAT
                

                existing_comments = set(
                    st.session_state
                    .all_comments["Komentar"]
                    .astype(str)
                    .tolist()
                )


                for message in messages:

                    if message in existing_comments:

                        continue

                    label, confidence = (
                        classify_comment(
                            message
                        )
                    )

                    new_row = pd.DataFrame(
                        [{
                            "Waktu": datetime.now(),
                            "Komentar": message,
                            "Prediksi": label,
                            "Confidence": (
                                confidence * 100
                            )
                        }]
                    )

                    st.session_state.all_comments = (
                        pd.concat(
                            [
                                st.session_state.all_comments,
                                new_row
                            ],
                            ignore_index=True
                        )
                    )

                    existing_comments.add(
                        message
                    )


                st_autorefresh(
                    interval=REFRESH_INTERVAL,
                    key="live_refresh"
                )

    # STATISTIK
   

    df_live = (
        st.session_state.all_comments
    )

    total = len(df_live)

    normal = (
        df_live["Prediksi"] == "Normal"
    ).sum()

    toxic = (
        df_live["Prediksi"] == "Toxic"
    ).sum()

    spam = (
        df_live["Prediksi"] == "Spam"
    ).sum()


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total",
        total
    )

    c2.metric(
        "🟢 Normal",
        normal
    )

    c3.metric(
        "🔴 Toxic",
        toxic
    )

    c4.metric(
        "🟡 Spam",
        spam
    )

    # CHAT TERAKHIR

    st.subheader(
        "Chat Terbaru"
    )

    for _, row in (
        df_live
        .tail(10)
        .iloc[::-1]
        .iterrows()
    ):

        if row["Prediksi"] == "Normal":

            css_class = "chat-normal"

        elif row["Prediksi"] == "Spam":

            css_class = "chat-spam"

        else:

            css_class = "chat-toxic"


        st.markdown(
            f"""
            <div class="chat-meta">
                {row["Waktu"].strftime("%H:%M:%S")}
                |
                {row["Prediksi"]}
                |
                Confidence:
                {row["Confidence"]:.2f}%
            </div>

            <div class="chat-bubble {css_class}">
                {row["Komentar"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# TAB 2 - DETEKSI MANUAL


with tab2:

    st.subheader(
        " Deteksi Komentar"
    )

    input_user = st.text_area(
        "Masukkan teks",
        height=120,
        placeholder=(
            "Contoh: terima kasih sudah live..."
        )
    )


    if st.button(
        " Analisis",
        type="primary",
        use_container_width=True
    ):

        label, confidence = (
            classify_comment(
                input_user
            )
        )


        st.session_state.manual_history.loc[
            len(st.session_state.manual_history)
        ] = [

            datetime.now(),

            input_user,

            label,

            confidence * 100
        ]

        # HASIL

        if label == "Normal":

            st.success(
                f"🟢 Hasil: **{label}**"
            )

            css_class = "chat-normal"


        elif label == "Spam":

            st.warning(
                f"🟡 Hasil: **{label}**"
            )

            css_class = "chat-spam"


        elif label == "Toxic":

            st.error(
                f"🔴 Hasil: **{label}**"
            )

            css_class = "chat-toxic"


        else:

            st.info(
                f"Hasil: {label}"
            )

            css_class = "chat-normal"


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        st.markdown(
            f"""
            <div class="chat-bubble {css_class}">
                {input_user}
            </div>
            """,
            unsafe_allow_html=True
        )

# TAB 3 - UPLOAD FILE

with tab3:

    st.subheader(
        "📁 Upload CSV / Excel"
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )


    if uploaded_file is not None:

        try:
            # BACA FILE

            if uploaded_file.name.lower().endswith(
                ".csv"
            ):

                df_upload = pd.read_csv(
                    uploaded_file
                )

            else:

                df_upload = pd.read_excel(
                    uploaded_file
                )


            st.success(
                f"File berhasil dibaca: "
                f"{len(df_upload)} data"
            )


            st.write(
                "### Preview Data"
            )

            st.dataframe(
                df_upload.head(10),
                use_container_width=True
            )
            # PILIH KOLOM

            text_column = st.selectbox(
                "Pilih kolom komentar/pesan",
                df_upload.columns
            )


            if st.button(
                " Analisis Data",
                type="primary"
            ):

                progress = st.progress(
                    0
                )

                hasil = []

                confidence_list = []

                total_data = len(
                    df_upload
                )

                # PREDIKSI

                for i, text in enumerate(
                    df_upload[text_column]
                    .fillna("")
                ):

                    label, confidence = (
                        classify_comment(
                            text
                        )
                    )

                    hasil.append(
                        label
                    )

                    confidence_list.append(
                        confidence * 100
                    )


                    progress.progress(
                        (i + 1)
                        / total_data
                    )

                # HASIL

                df_upload["Prediksi"] = (
                    hasil
                )

                df_upload["Confidence"] = (
                    confidence_list
                )


                st.success(
                    "Analisis selesai!"
                )


                st.write(
                    "### Hasil Analisis"
                )

                st.dataframe(
                    df_upload,
                    use_container_width=True
                )

                # STATISTIK

                normal = (
                    df_upload["Prediksi"]
                    == "Normal"
                ).sum()

                toxic = (
                    df_upload["Prediksi"]
                    == "Toxic"
                ).sum()

                spam = (
                    df_upload["Prediksi"]
                    == "Spam"
                ).sum()


                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "🟢 Normal",
                    normal
                )

                c2.metric(
                    "🔴 Toxic",
                    toxic
                )

                c3.metric(
                    "🟡 Spam",
                    spam
                )

                # CHART

                chart = pd.DataFrame(
                    {
                        "Jumlah": [
                            normal,
                            toxic,
                            spam
                        ]
                    },
                    index=[
                        "Normal",
                        "Toxic",
                        "Spam"
                    ]
                )

                st.bar_chart(
                    chart
                )

                # DOWNLOAD


                csv_result = (
                    df_upload
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8-sig")
                )


                st.download_button(
                    "Download Hasil CSV",
                    csv_result,
                    "hasil_prediksi.csv",
                    "text/csv"
                )


        except Exception as e:

            st.error(
                f"Gagal membaca file: {e}"
            )


# =========================================================
# SUMMARY
# =========================================================

def show_summary(
    df,
    title
):

    if df.empty:

        return


    st.divider()

    st.subheader(
        f" Kesimpulan Analisis {title}"
    )


    total = len(df)

    normal = (
        df["Prediksi"] == "Normal"
    ).sum()

    toxic = (
        df["Prediksi"] == "Toxic"
    ).sum()

    spam = (
        df["Prediksi"] == "Spam"
    ).sum()


    p_toxic = (
        toxic / total * 100
        if total > 0
        else 0
    )

    p_spam = (
        spam / total * 100
        if total > 0
        else 0
    )


    c1, c2 = st.columns(2)


    with c1:

        st.write(
            f"🟢 Normal: **{normal}**"
        )

        st.write(
            f"🔴 Toxic: **{toxic}**"
        )

        st.write(
            f"🟡 Spam: **{spam}**"
        )


    with c2:

        chart = pd.DataFrame(
            {
                "Jumlah": [
                    normal,
                    toxic,
                    spam
                ]
            },
            index=[
                "Normal",
                "Toxic",
                "Spam"
            ]
        )

        st.bar_chart(
            chart
        )


    if title == "Live Chat":

        if (
            p_toxic > 20
            or
            (p_spam + p_toxic) > 40
        ):

            st.error(
                """
                **KONDISI BAHAYA**

                Persentase Spam dan Toxic cukup tinggi.

                **Rekomendasi:**
                - Aktifkan Slow Mode
                - Gunakan moderator
                - Pantau akun yang berulang kali melakukan pelanggaran
                - Batasi pengiriman pesan berulang
                - Nonaktifkan link di live chat sementara
                """
            )


        elif (
            p_toxic > 5
            or
            p_spam > 15
        ):

            st.warning(
                """
                 **KONDISI WASPADA**

                Terdapat peningkatan pesan Spam dan Toxic.
                namun masih dalam batas yang dapat dikendalikan
                **Rekomendasi:**
                - Pantau Live Chat secara berkala
                - Bukan Tontonan anak - anak dibawah umur
                - Siapkan moderator
                - Batasi pengiriman pesan beruntun
                """
            )


        else:

            st.success(
                """
                 **KONDISI AMAN**

                Live Chat didominasi pesan positif dan relevan.
                **Rekomendasi
                - Live chat dapat berjalan normal"
                - Tetap lakukan pemantauan rutin"
                """
            )



# TAMPILKAN SUMMARY


with tab1:

    if not st.session_state.is_running:

        show_summary(
            st.session_state.all_comments,
            "Live Chat"
        )


with tab2:

    show_summary(
        st.session_state.manual_history,
        "Deteksi Manual"
    )