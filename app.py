import os
import io
import warnings
import time
import threading
import uuid
import requests
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
from scipy.stats import rankdata
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

warnings.filterwarnings("ignore")

# --- 1. inisialisasi app & cors ---
app = FastAPI(title="churnshield api - visions project", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# menentukan direktori utama tempat app.py berada
base_dir = os.path.dirname(os.path.abspath(__file__))

# load .env dari direktori saat ini atau folder ML PBL
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(os.path.join(base_dir, 'ML PBL', '.env'))

# --- 2. load semua alat tempur ---
def load_pkl(filename):
    # Coba direktori utama tempat app.py berada
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[WARN] Gagal load {path}: {e}")
    # Coba di folder ML PBL
    path_sub = os.path.join(base_dir, 'ML PBL', filename)
    if os.path.exists(path_sub):
        try:
            return joblib.load(path_sub)
        except Exception as e:
            print(f"[WARN] Gagal load {path_sub}: {e}")
    raise FileNotFoundError(f"File {filename} tidak ditemukan di root maupun folder ML PBL")

try:
    model = load_pkl('model_churn_final.pkl')
    scaler = load_pkl('scaler.pkl')
    kmeans = load_pkl('kmeans.pkl')
    model_columns = load_pkl('model_columns.pkl')
    print("[OK] sukses load semua model pkl!")
except Exception as e:
    print(f"[ERROR] gagal load pkl: {e}")
    model_columns = []

# --- 3. fungsi anti-bocor ---
def filter_masa_depan(df_transaksi, kolom_tanggal, df_churn_date):
    df_temp = df_transaksi.merge(df_churn_date, on='customer_id', how='left')
    df_temp[kolom_tanggal] = pd.to_datetime(df_temp[kolom_tanggal], format='mixed', errors='coerce', dayfirst=True)
    df_temp['unsubscribed_date'] = pd.to_datetime(df_temp['unsubscribed_date'], format='mixed', errors='coerce', dayfirst=True)
    
    kondisi_valid = (df_temp[kolom_tanggal] <= df_temp['unsubscribed_date']) | (df_temp['unsubscribed_date'].isna())
    return df_temp[kondisi_valid].copy()

# --- 4. fungsi hitung severity ---
def hitung_severity(row):
    skor = 0
    if row['priority'] in ['High', 'Critical']: skor += 3
    elif row['priority'] == 'Medium': skor += 2
    else: skor += 1
    
    if row['status'] == 'Open': skor += 2
    return skor

# Path untuk CSV lokal
CSV_PATH = os.path.join(base_dir, "churn_prediction_results.csv")
OUTPUT_CSV = os.path.join(base_dir, "churnshield_complete_output.csv")

# State penampung dataframe
_df = None

# --- 5. endpoint utama ---
@app.post("/predict-batch")
async def predict_batch(
    file_accounts: UploadFile = File(...),
    file_usage: UploadFile = File(...),
    file_billing: UploadFile = File(...),
    file_tickets: UploadFile = File(...),
    file_nps: UploadFile = File(...)
):
    global _df
    if not model_columns:
        raise HTTPException(status_code=500, detail="model_columns.pkl gagal dimuat oleh server")

    try:
        # baca semua file mentah
        df_accounts = pd.read_csv(io.BytesIO(await file_accounts.read()))
        df_usage = pd.read_csv(io.BytesIO(await file_usage.read()))
        df_billing = pd.read_csv(io.BytesIO(await file_billing.read()))
        df_tickets = pd.read_csv(io.BytesIO(await file_tickets.read()))
        df_nps = pd.read_csv(io.BytesIO(await file_nps.read()))

        # fix: contract_type dibiarin kapital kayak aslinya di dataset
        df_accounts['plan_type'] = df_accounts['plan_type'].astype(str).str.strip().str.lower()
        df_accounts['contract_type'] = df_accounts['contract_type'].astype(str).str.strip()
        df_accounts['churn'] = df_accounts['unsubscribed_date'].notna().astype(int)
        
        df_churn_date = df_accounts[['customer_id', 'unsubscribed_date']]

        # filter masa depan
        df_usage_v = filter_masa_depan(df_usage, 'last_login_date', df_churn_date)
        df_tickets_v = filter_masa_depan(df_tickets, 'created_date', df_churn_date)
        df_billing_v = filter_masa_depan(df_billing, 'payment_date', df_churn_date)
        df_nps_v = filter_masa_depan(df_nps, 'survey_date', df_churn_date)

        # proses data billing
        df_billing_v['billing_date'] = pd.to_datetime(df_billing_v['billing_date'], format='mixed', dayfirst=True, errors='coerce')
        df_billing_v['payment_date'] = pd.to_datetime(df_billing_v['payment_date'], format='mixed', dayfirst=True, errors='coerce')
        df_billing_v['delay_hari'] = (df_billing_v['payment_date'] - df_billing_v['billing_date']).dt.days
        
        df_pay = df_billing_v[df_billing_v['record_type']=='payment'].groupby('customer_id').agg(
            total_payment_value=('payment_value', 'sum'),
            avg_payment_delay=('delay_hari', 'mean')
        ).reset_index()
        
        df_dun = df_billing_v[df_billing_v['record_type']=='dunning'].groupby('customer_id').size().reset_index(name='total_dunning')

        # proses data usage
        df_usage_agg = df_usage_v.groupby('customer_id').agg(
            avg_usage_hrs=('monthly_usage_hrs', 'mean'),
            avg_feature_adoption=('feature_adoption_pct', 'mean'),
            last_login=('last_login_date', 'max')
        ).reset_index()

        # proses data nps
        df_nps_agg = df_nps_v.groupby('customer_id').agg(avg_nps_score=('nps_score', 'mean')).reset_index()

        # proses data tiket
        if not df_tickets_v.empty:
            df_tickets_v['sev_score'] = df_tickets_v.apply(hitung_severity, axis=1)
            df_tkt_agg = df_tickets_v.groupby('customer_id').agg(
                total_tickets=('ticket_id', 'count'),
                avg_severity=('sev_score', 'mean'),
                severe_tkt_count=('sev_score', lambda x: (x >= 4).sum())
            ).reset_index()
            df_tkt_agg['severe_ticket_ratio'] = df_tkt_agg['severe_tkt_count'] / df_tkt_agg['total_tickets'].replace(0,1)
        else:
            df_tkt_agg = pd.DataFrame(columns=['customer_id', 'total_tickets', 'avg_severity', 'severe_ticket_ratio'])

        # gabungin semua tabel
        df_master = df_accounts.copy()
        df_master = df_master.merge(df_usage_agg, on='customer_id', how='left')\
                             .merge(df_tkt_agg, on='customer_id', how='left')\
                             .merge(df_nps_agg, on='customer_id', how='left')\
                             .merge(df_pay, on='customer_id', how='left')\
                             .merge(df_dun, on='customer_id', how='left')

        # hitung tanggal persis kayak eda
        df_master['subscription_date'] = pd.to_datetime(df_master['subscription_date'], format='mixed', dayfirst=True, errors='coerce')
        df_master['unsubscribed_date'] = pd.to_datetime(df_master['unsubscribed_date'], format='mixed', dayfirst=True, errors='coerce')
        df_master['last_login'] = pd.to_datetime(df_master['last_login'], format='mixed', dayfirst=True, errors='coerce')
        
        tanggal_acuan = pd.to_datetime('2024-12-31')
        df_master['end_date'] = df_master['unsubscribed_date'].fillna(tanggal_acuan)
        
        df_master['tenure_months'] = ((df_master['end_date'] - df_master['subscription_date']).dt.days / 30).round(1)
        df_master['days_since_login'] = (df_master['end_date'] - df_master['last_login']).dt.days
        df_master['usage_per_user'] = df_master['avg_usage_hrs'] / df_master['total_users'].replace(0,1)

        # =======================================================
        # proses fillna (imputasi)
        # =======================================================
        cols_fill_zero = ['total_tickets', 'total_dunning', 'avg_payment_delay', 'total_payment_value', 'avg_severity', 'severe_ticket_ratio']
        for c in cols_fill_zero:
            if c in df_master.columns:
                df_master[c] = df_master[c].fillna(0.0)

        cols_fill_median = ['avg_nps_score', 'avg_usage_hrs', 'avg_feature_adoption', 'usage_per_user', 'days_since_login']
        for c in cols_fill_median:
            if c in df_master.columns and not df_master[c].isna().all():
                df_master[c] = df_master[c].fillna(df_master[c].median())
            else:
                df_master[c] = df_master[c].fillna(0.0)

        # tipe customer
        def get_cust_type(row):
            if row.get('total_tickets', 0) == 0: return 'loyal_quiet'
            if row.get('avg_severity', 0) >= 4 or row.get('severe_ticket_ratio', 0) > 0.5: return 'problematic'
            if row.get('avg_severity', 0) >= 3 or row.get('severe_ticket_ratio', 0) > 0.3: return 'at_risk'
            return 'satisfied'
        
        df_master['customer_type'] = df_master.apply(get_cust_type, axis=1)

        # persiapkan kolom untuk dimasukkan ke machine learning
        df_model_input = df_master.drop(columns=['customer_id','subscription_date','unsubscribed_date','last_login','end_date','churn'], errors='ignore')
        
        df_model_input = pd.get_dummies(df_model_input, columns=['plan_type','contract_type','customer_type'], drop_first=True)
        
        # fix: ubah boolean jadi integer
        for col in df_model_input.select_dtypes(include=['bool']).columns:
            df_model_input[col] = df_model_input[col].astype(int)

        # paksa kolom biar urutannya persis kayak waktu training
        for col in model_columns:
            if col not in df_model_input.columns: 
                df_model_input[col] = 0
        df_model_input = df_model_input[model_columns]

        # proses scalling & k-means
        X_scaled = scaler.transform(df_model_input)
        df_scaled = pd.DataFrame(X_scaled, columns=model_columns)
        df_scaled['customer_segment'] = kmeans.predict(df_scaled)
        
        # =======================================================
        # eksekusi prediksi akhir menggunakan ranking (sama seperti jupyter)
        # =======================================================
        # 1. ambil probabilitas mentah (index 1)
        probs = model.predict_proba(df_scaled)[:, 1]
        
        # 2. hitung persentil berbasis ranking
        ranks = rankdata(probs)
        if len(ranks) > 1:
            churn_score_ranked = ((ranks - 1) / (len(ranks) - 1) * 100).round(1)
        else:
            churn_score_ranked = pd.Series([0.0]) # handle jika datanya cuma 1 baris
            
        df_master['churn_score'] = churn_score_ranked
        
        # 3. tentukan level risiko
        df_master['risk_level'] = df_master['churn_score'].apply(
            lambda x: 'High' if x >= 70 else ('Medium' if x >= 30 else 'Low')
        )

        # ngerapihin kapital data buat dibalikin ke frontend
        df_master['plan_type'] = df_master['plan_type'].astype(str).str.title()
        df_master['contract_type'] = df_master['contract_type'].astype(str).str.title()
        df_master['customer_type'] = df_master['customer_type'].astype(str).str.replace('_', ' ').str.title()
        df_master['mrr'] = (df_master['total_payment_value'] / df_master['tenure_months'].replace(0,1)).round(2)
        
        # --- daftar lengkap kolom termasuk churn & score ---
        output_cols = [
            'customer_id', 'plan_type', 'contract_type', 'customer_type', 'tenure_months', 
            'total_payment_value', 'mrr', 'total_dunning', 'avg_payment_delay', 'days_since_login', 
            'avg_nps_score', 'total_tickets', 'avg_severity', 'severe_ticket_ratio', 
            'avg_usage_hrs', 'usage_per_user', 'avg_feature_adoption', 
            'churn', 'churn_score', 'risk_level'
        ]
        valid_cols = [c for c in output_cols if c in df_master.columns]
        
        # simpan ke state global _df
        _df = df_master[valid_cols].copy()
        if "risk_level" in _df.columns:
            _df["risk_level"] = _df["risk_level"].astype(str).str.capitalize()
        _df = _df.fillna(0)
        
        # simpan CSV lokal
        try:
            _df.to_csv(CSV_PATH, index=False)
            print(f"[Local] {len(_df)} pelanggan disimpan ke {CSV_PATH}")
        except Exception as e:
            print(f"[Local] GAGAL simpan CSV: {e}")

        # Supabase Sync
        sb_url = os.getenv("SUPABASE_URL", "")
        sb_key = os.getenv("SUPABASE_KEY", "")
        if sb_url and sb_key:
            try:
                records = []
                for _, row in df_master[valid_cols].iterrows():
                    records.append({
                        "customer_id":          str(row["customer_id"]),
                        "plan_type":            str(row.get("plan_type", "")),
                        "contract_type":        str(row.get("contract_type", "")),
                        "customer_type":        str(row.get("customer_type", "")),
                        "tenure_months":        float(row.get("tenure_months", 0)),
                        "total_payment_value":  float(row.get("total_payment_value", 0)),
                        "mrr":                  float(row.get("mrr", 0)),
                        "total_dunning":        int(row.get("total_dunning", 0)),
                        "avg_payment_delay":    float(row.get("avg_payment_delay", 0)),
                        "days_since_login":     int(row.get("days_since_login", 0)),
                        "avg_nps_score":        float(row.get("avg_nps_score", 0)),
                        "total_tickets":        int(row.get("total_tickets", 0)),
                        "avg_severity":         float(row.get("avg_severity", 0)),
                        "severe_ticket_ratio":  float(row.get("severe_ticket_ratio", 0)),
                        "avg_usage_hrs":        float(row.get("avg_usage_hrs", 0)),
                        "usage_per_user":       float(row.get("usage_per_user", 0)),
                        "avg_feature_adoption": float(row.get("avg_feature_adoption", 0)),
                        "churn_actual":         bool(row.get("churn", False)),
                        "churn_score":          float(row.get("churn_score", 0)),
                        "risk_level":           str(row.get("risk_level", "Low")),
                    })
                hdrs = {
                    "apikey": sb_key,
                    "Authorization": "Bearer " + sb_key,
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                }
                for i in range(0, len(records), 500):
                    r = requests.post(
                        sb_url.rstrip("/") + "/rest/v1/customers",
                        headers=hdrs,
                        json=records[i:i+500],
                        timeout=60,
                    )
                    if r.status_code not in (200, 201):
                        print(f"[Supabase] upsert batch {i//500+1} gagal HTTP {r.status_code}: {r.text[:200]}")
                print(f"[Supabase] {len(records)} pelanggan disimpan ke tabel customers")
            except Exception as e:
                print(f"[Supabase] gagal simpan: {e}")

        output = io.StringIO()
        df_master[valid_cols].to_csv(output, index=False)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=churn_prediction_results.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================================================
# --- CHATBOT & LLM SECTION (v2 — Deep NLP Engine) ---
# =======================================================

GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = "llama-3.1-8b-instant"        # model cepat untuk chat & batch
GROQ_MODEL_DEEP = "llama-3.3-70b-versatile"     # model powerful untuk analisis mendalam

# -------------------------------------------------------
# SYSTEM PROMPTS — Expert Persona dengan Deep Framework
# -------------------------------------------------------

SYSTEM_PROMPT = """Kamu adalah ChurnShield AI, sistem kecerdasan buatan khusus retensi pelanggan SaaS milik LapisAI.

IDENTITAS & KEAHLIAN:
Kamu adalah Customer Success Expert dengan pengalaman 10+ tahun di industri SaaS B2B. Kamu memahami secara mendalam:
- Psikologi pelanggan SaaS: mengapa mereka bertahan dan mengapa mereka pergi
- Pola churn: sudden churn, gradual disengagement, competitive churn, value misalignment
- Framework retensi: QBR, health scoring, early warning system, playbook eskalasi
- Metrik SaaS: MRR, NPS, DAU/MAU, feature adoption, time-to-value

PRINSIP ANALISIS:
1. Baca sinyal perilaku, bukan hanya angka — angka adalah gejala, cari penyebabnya
2. Setiap pelanggan memiliki konteks unik — hindari saran generik
3. Tindakan harus SMART: Specific, Measurable, Actionable, Relevant, Time-bound
4. Prioritas: selamatkan revenue dulu, baru bangun relasi jangka panjang

FORMAT OUTPUT:
Tulis dalam 3 paragraf mengalir, tanpa header atau bullet point:
- Paragraf 1: Kondisi pelanggan saat ini — ceritakan situasi dengan empati dan ketepatan, sebutkan metrik kunci yang paling mengkhawatirkan
- Paragraf 2: Akar masalah dan dinamika — jelaskan MENGAPA ini terjadi, bukan hanya APA yang terjadi, hubungkan pola perilaku dengan kebutuhan yang tidak terpenuhi
- Paragraf 3: Rencana tindakan konkret — 3 langkah dengan siapa yang bertanggung jawab, kapan dilakukan, dan indikator keberhasilan

LARANGAN KERAS:
- Jangan tulis label seperti "Paragraf 1:", "Kondisi:", "Akar Masalah:", atau header apapun
- Jangan berikan saran generik seperti "tingkatkan engagement" tanpa konteks spesifik
- Jangan mengarang data yang tidak ada di input
- Jangan menggunakan bullet point
"""

SYSTEM_PROMPT_EN = """You are ChurnShield AI, LapisAI's specialized artificial intelligence system for SaaS customer retention.

IDENTITY & EXPERTISE:
You are a Customer Success Expert with 10+ years of experience in the B2B SaaS industry. You deeply understand:
- SaaS customer psychology: why they stay and why they leave
- Churn patterns: sudden churn, gradual disengagement, competitive churn, value misalignment
- Retention frameworks: QBR, health scoring, early warning system, escalation playbooks
- SaaS metrics: MRR, NPS, DAU/MAU, feature adoption, time-to-value

ANALYSIS PRINCIPLES:
1. Read behavioral signals, not just numbers — numbers are symptoms, find the root cause
2. Every customer has a unique context — avoid generic advice
3. Actions must be SMART: Specific, Measurable, Actionable, Relevant, Time-bound
4. Priority: save revenue first, then build long-term relationships

OUTPUT FORMAT:
Write in 3 flowing paragraphs, without headers or bullet points:
- Paragraph 1: Current customer condition — describe the situation with empathy and precision, mentioning the most concerning key metrics
- Paragraph 2: Root cause and dynamics — explain WHY this is happening, not just WHAT is happening, connecting behavior patterns with unmet needs
- Paragraph 3: Concrete action plan — 3 steps detailing who is responsible, when it should be done, and success indicators

STRICT PROHIBITIONS:
- Do NOT write labels like "Paragraph 1:", "Condition:", "Root Cause:", or any headers
- Do NOT provide generic advice like "increase engagement" without specific context
- Do NOT invent data that is not present in the input
- Do NOT use bullet points
"""

CHAT_SYSTEM_PROMPT = """Kamu adalah ChurnShield AI, asisten strategis Customer Success untuk platform SaaS LapisAI.

KEMAMPUAN UTAMA:
1. ANALISIS FAKTUAL: Menjawab pertanyaan tentang data pelanggan dengan presisi — siapa, berapa, kapan
2. ANALISIS MENDALAM: Mengidentifikasi pola, tren, dan anomali tersembunyi dalam data churn
3. STRATEGI RETENSI: Memberikan rekomendasi berbasis best practice Customer Success
4. PERBANDINGAN SEGMEN: Membandingkan performa antar plan_type, contract_type, customer_type

CARA MENJAWAB BERDASARKAN JENIS PERTANYAAN:
- Pertanyaan faktual ("siapa", "berapa", "mana"): Jawab LANGSUNG dengan data. Sebutkan Customer ID. Singkat dan padat.
- Pertanyaan analitis ("mengapa", "bagaimana pola"): Berikan analisis dengan insight yang tidak obvious
- Pertanyaan strategis ("apa yang harus dilakukan"): Framework + tindakan prioritas + timeline
- Pertanyaan perbandingan: Gunakan format terstruktur dengan angka konkret

PRINSIP RESPONS:
- Selalu dasarkan jawaban pada DATA AKTUAL yang diberikan, bukan pengetahuan umum
- Jika kamu mendeteksi anomali atau insight penting yang tidak ditanya, sebutkan sebagai "catatan penting"
- Gunakan Bahasa Indonesia profesional namun conversational
- Jangan mengulang pertanyaan pengguna di awal jawaban
"""

CHAT_SYSTEM_PROMPT_EN = """You are ChurnShield AI, a strategic Customer Success assistant for the LapisAI SaaS platform.

CORE CAPABILITIES:
1. FACTUAL ANALYSIS: Answer questions about customer data with precision — who, how much, when
2. DEEP ANALYSIS: Identify hidden patterns, trends, and anomalies in churn data
3. RETENTION STRATEGY: Provide recommendations based on Customer Success best practices
4. SEGMENT COMPARISON: Compare performance across plan_type, contract_type, customer_type

HOW TO ANSWER BASED ON QUESTION TYPE:
- Factual questions ("who", "how many", "which"): Answer DIRECTLY with data. Mention specific Customer ID. Short and concise.
- Analytical questions ("why", "how pattern"): Provide analysis with non-obvious insights
- Strategic questions ("what should be done"): Framework + action priorities + timeline
- Comparison questions: Use a structured format with concrete numbers

RESPONSE PRINCIPLES:
- Always base your answers on the ACTUAL DATA provided, not general knowledge
- If you detect a significant anomaly or insight not directly asked, mention it as an "important note"
- Use professional yet conversational English
- Do not repeat the user's question at the beginning of your response
"""

# -------------------------------------------------------
# STATE
# -------------------------------------------------------
_retensi_progress = {"running": False, "done": 0, "total": 0, "status": "idle"}

# Conversation history store: {session_id: [{"role": ..., "content": ...}, ...]}
_chat_sessions: dict = {}
CHAT_MAX_HISTORY = 10  # simpan max N pesan terakhir per sesi


# -------------------------------------------------------
# SUPABASE HELPERS
# -------------------------------------------------------
def get_supabase():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return {"url": url.rstrip("/"), "key": key}


def _sb_headers(cfg):
    return {
        "apikey": cfg["key"],
        "Authorization": "Bearer " + cfg["key"],
        "Content-Type": "application/json",
    }


def _load_from_supabase():
    cfg = get_supabase()
    if cfg is None:
        return None
    try:
        rows, offset, limit = [], 0, 1000
        while True:
            r = requests.get(
                cfg["url"] + "/rest/v1/customers",
                headers={**_sb_headers(cfg), "Range-Unit": "items", "Range": f"{offset}-{offset+limit-1}"},
                params={"select": "*"},
                timeout=30,
            )
            if r.status_code not in (200, 206):
                print(f"[Supabase] HTTP {r.status_code}: {r.text[:200]}")
                return None
            data = r.json()
            rows.extend(data)
            if len(data) < limit:
                break
            offset += limit
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"[Supabase] error: {e}")
        return None


def load_data():
    df = _load_from_supabase()
    if df is None:
        try:
            df = pd.read_csv(CSV_PATH)
        except FileNotFoundError:
            return None
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].astype(str).str.capitalize()
    return df.fillna(0)


def get_df():
    global _df
    if _df is None:
        _df = load_data()
        if _df is None:
            raise HTTPException(status_code=503, detail="Data tidak tersedia. Jalankan /predict-batch dulu.")
    return _df


def _buat_notifikasi(df_high: pd.DataFrame):
    cfg = get_supabase()
    if cfg is None:
        return
    try:
        customer_ids = df_high["customer_id"].tolist()
        hdrs = _sb_headers(cfg)
        requests.delete(
            cfg["url"] + "/rest/v1/notifikasi",
            headers=hdrs,
            params={"customer_id": "in.(" + ",".join(str(x) for x in customer_ids) + ")", "dibaca": "eq.false"},
            timeout=30,
        )
        records = [
            {
                "customer_id": row["customer_id"],
                "pesan": (
                    f"High Risk — {row['customer_id']} | "
                    f"Score: {round(float(row['churn_score']), 1)} | "
                    f"{row.get('kategori_tindakan', 'Perlu tindakan segera')}"
                ),
                "tipe": "warning",
                "dibaca": False,
            }
            for _, row in df_high.iterrows()
        ]
        for i in range(0, len(records), 500):
            requests.post(
                cfg["url"] + "/rest/v1/notifikasi",
                headers=hdrs,
                json=records[i:i+500],
                timeout=30,
            )
    except Exception as e:
        print(f"[notifikasi] error: {e}")


# -------------------------------------------------------
# LLM CALL HELPERS
# -------------------------------------------------------
def chat_llm(messages, max_tokens=600, model=None):
    """Panggil LLM untuk single request (rekomendasi, analisis, chat)."""
    api_key = os.getenv("GROQ_API_KEY", "")
    used_model = model or GROQ_MODEL
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={"model": used_model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.65},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            return "Error: Rate limit Groq. Tunggu 1 menit lalu coba lagi."
        return "Error: " + str(r.status_code)
    except requests.exceptions.Timeout:
        return "Timeout, coba lagi"
    except Exception as e:
        return "Error: " + str(e)


def _get_api_keys():
    """Mengambil semua GROQ_API_KEY secara dinamis (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... dst tanpa batasan)."""
    keys = []
    # Coba key utama
    if key := os.getenv("GROQ_API_KEY"):
        keys.append(key)
    # Coba key berikutnya secara berurutan (2, 3, 4, ...)
    i = 2
    while True:
        key_next = os.getenv(f"GROQ_API_KEY_{i}")
        if not key_next:
            break
        keys.append(key_next)
        i += 1
    return keys


def chat_llm_batch(messages, max_tokens=700, _idx=[0]):
    """Round-robin multi-key untuk batch processing besar."""
    keys = _get_api_keys()
    if not keys:
        return "Error: tidak ada GROQ_API_KEY di .env"
    for attempt in range(len(keys) * 3):
        key = keys[_idx[0] % len(keys)]
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.65},
                timeout=60,
            )
            if r.status_code == 200:
                _idx[0] += 1
                return r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                _idx[0] += 1
                if attempt >= len(keys) - 1:
                    time.sleep(65)
                continue
            return "Error: " + str(r.status_code)
        except requests.exceptions.Timeout:
            time.sleep(10)
            continue
        except Exception as e:
            return "Error: " + str(e)
    return "Error: semua key kena rate limit"


def test_groq():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return False
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False


# -------------------------------------------------------
# DATASET BENCHMARKING — konteks relatif untuk LLM
# -------------------------------------------------------
def get_dataset_benchmarks(df: pd.DataFrame) -> dict:
    """
    Hitung statistik agregat dataset untuk dijadikan konteks perbandingan
    saat LLM menganalisis seorang pelanggan.
    """
    metrics = ['churn_score', 'avg_nps_score', 'avg_usage_hrs', 'avg_feature_adoption',
               'days_since_login', 'total_dunning', 'avg_payment_delay', 'total_tickets']
    benchmarks = {}
    for m in metrics:
        if m in df.columns:
            col = pd.to_numeric(df[m], errors='coerce').dropna()
            benchmarks[m] = {
                "median": round(float(col.median()), 2),
                "p25":    round(float(col.quantile(0.25)), 2),
                "p75":    round(float(col.quantile(0.75)), 2),
                "mean":   round(float(col.mean()), 2),
            }
    return benchmarks


def format_benchmarks_text(row, benchmarks: dict, lang="id") -> str:
    """Bandingkan metrik pelanggan vs. benchmark dataset, return string deskriptif."""
    lines = []
    is_en = lang.lower() == "en"
    comparisons = {
        "churn_score":         ("Churn Score",         True),   # True = nilai tinggi = buruk
        "avg_nps_score":       ("NPS Score",           False),  # False = nilai tinggi = baik
        "avg_usage_hrs":       ("Usage (jam/bln)" if not is_en else "Usage (hrs/mo)",     False),
        "avg_feature_adoption":("Feature Adoption",    False),
        "days_since_login":    ("Hari Sejak Login" if not is_en else "Days Since Login",    True),
        "total_dunning":       ("Jumlah Dunning" if not is_en else "Total Dunning",      True),
        "avg_payment_delay":   ("Rata-rata Delay Bayar" if not is_en else "Avg Payment Delay",True),
        "total_tickets":       ("Total Tiket" if not is_en else "Total Tickets",         True),
    }
    for key, (label, higher_is_worse) in comparisons.items():
        if key not in benchmarks or key not in row:
            continue
        val     = float(row[key])
        median  = benchmarks[key]["median"]
        p25     = benchmarks[key]["p25"]
        p75     = benchmarks[key]["p75"]
        if higher_is_worse:
            if val >= p75:   posisi = "jauh LEBIH BURUK dari rata-rata (top 25% terburuk)" if not is_en else "far WORSE than average (top 25% worst)"
            elif val >= median: posisi = "sedikit di atas rata-rata" if not is_en else "slightly above average"
            elif val >= p25: posisi = "di bawah rata-rata (kondisi baik)" if not is_en else "below average (good condition)"
            else:            posisi = "jauh DI BAWAH rata-rata (kondisi sangat baik)" if not is_en else "far BELOW average (very good condition)"
        else:
            if val >= p75:   posisi = "jauh DI ATAS rata-rata (top 25% terbaik)" if not is_en else "far ABOVE average (top 25% best)"
            elif val >= median: posisi = "di atas rata-rata" if not is_en else "above average"
            elif val >= p25: posisi = "sedikit di bawah rata-rata" if not is_en else "slightly below average"
            else:            posisi = "jauh LEBIH RENDAH dari rata-rata (perlu perhatian)" if not is_en else "far LOWER than average (needs attention)"
        
        if is_en:
            lines.append(f"  - {label}: {round(val, 1)} → {posisi} (dataset median: {median})")
        else:
            lines.append(f"  - {label}: {round(val, 1)} → {posisi} (median dataset: {median})")
    return "\n".join(lines)


def detect_anomaly(row, df: pd.DataFrame, lang="id") -> list:
    """
    Deteksi sinyal anomali yang tidak obvious dari angka mentah.
    Return list string deskripsi anomali.
    """
    anomalies = []
    benchmarks = get_dataset_benchmarks(df)
    is_en = lang.lower() == "en"

    # NPS jauh di bawah rata-rata plan type-nya
    if "avg_nps_score" in row and "plan_type" in row:
        plan_nps = df[df["plan_type"] == row["plan_type"]]["avg_nps_score"]
        if len(plan_nps) > 3:
            plan_median = float(plan_nps.median())
            if float(row["avg_nps_score"]) < plan_median * 0.6:
                if is_en:
                    anomalies.append(
                        f"Customer NPS ({round(float(row['avg_nps_score']),1)}) is far below the average "
                        f"of other {row['plan_type']} users ({round(plan_median,1)}) — significant anomaly"
                    )
                else:
                    anomalies.append(
                        f"NPS pelanggan ({round(float(row['avg_nps_score']),1)}) jauh di bawah rata-rata "
                        f"pengguna {row['plan_type']} lainnya ({round(plan_median,1)}) — anomali signifikan"
                    )

    # Tenure lama tapi churn score tinggi — veteran yang mulai kabur
    tenure = float(row.get("tenure_months", 0))
    churn  = float(row.get("churn_score", 0))
    if tenure >= 18 and churn >= 65:
        if is_en:
            anomalies.append(
                f"Veteran customer ({round(tenure,0)} months) with high churn score ({round(churn,1)}) — "
                "pattern of 'veteran churn' often triggered by internal changes or new competitors"
            )
        else:
            anomalies.append(
                f"Pelanggan veteran ({round(tenure,0)} bulan) dengan churn score tinggi ({round(churn,1)}) — "
                "pola 'veteran churn' yang seringkali dipicu perubahan internal atau kompetitor baru"
            )

    # Usage tinggi tapi NPS rendah — menggunakan produk tapi tidak puas
    usage = float(row.get("avg_usage_hrs", 0))
    nps   = float(row.get("avg_nps_score", 0))
    if usage >= 50 and nps <= 5:
        if is_en:
            anomalies.append(
                f"Active user ({round(usage,1)} hrs/mo) with very low NPS ({round(nps,1)}) — "
                "sign of trapped customer: forced to use but unsatisfied, highly vulnerable to competitors"
            )
        else:
            anomalies.append(
                f"Pengguna aktif ({round(usage,1)} jam/bln) dengan NPS sangat rendah ({round(nps,1)}) — "
                "tanda trapped customer: terpaksa pakai tapi tidak puas, sangat rentan ke kompetitor"
            )

    # Banyak tiket severity tinggi tapi masih aktif — menunggu bom waktu
    sev_ratio = float(row.get("severe_ticket_ratio", 0))
    tickets   = float(row.get("total_tickets", 0))
    if sev_ratio > 0.5 and tickets >= 3:
        if is_en:
            anomalies.append(
                f"{round(sev_ratio*100,0)}% of tickets have severe severity out of {int(tickets)} total tickets — "
                "accumulation of unresolved frustrations"
            )
        else:
            anomalies.append(
                f"{round(sev_ratio*100,0)}% tiket berstatus parah dari {int(tickets)} tiket total — "
                "akumulasi frustrasi yang belum terselesaikan"
            )

    # MRR tinggi tapi dunning ada — risiko revenue loss besar
    mrr     = float(row.get("mrr", 0))
    dunning = float(row.get("total_dunning", 0))
    if mrr > 0 and dunning >= 2:
        if "mrr" in benchmarks:
            if mrr >= benchmarks["mrr"]["p75"]:
                if is_en:
                    anomalies.append(
                        f"High MRR customer (Rp {round(mrr,0):,.0f}) has {int(dunning)} dunning instances — "
                        "risk of significant revenue loss"
                    )
                else:
                    anomalies.append(
                        f"Pelanggan dengan MRR tinggi (Rp {round(mrr,0):,.0f}) memiliki {int(dunning)} dunning — "
                        "risiko kehilangan revenue signifikan"
                    )

    return anomalies


def context_pelanggan(row, lang="id") -> str:
    is_en = lang.lower() == "en"
    if is_en:
        return (
            "Customer data:\n"
            "- ID             : " + str(row["customer_id"]) + "\n"
            "- Plan           : " + str(row["plan_type"]) + "\n"
            "- Contract       : " + str(row.get("contract_type", "-")) + "\n"
            "- Customer Type  : " + str(row.get("customer_type", "-")) + "\n"
            "- Tenure         : " + str(round(float(row["tenure_months"]), 1)) + " months\n"
            "- MRR            : Rp " + str(round(float(row.get("mrr", 0)), 2)) + "\n"
            "- Total Payment  : Rp " + str(round(float(row.get("total_payment_value", 0)), 2)) + "\n"
            "- Churn Score    : " + str(round(float(row["churn_score"]), 1)) + "/100\n"
            "- Risk Level     : " + str(row["risk_level"]) + "\n"
            "- Dunning        : " + str(int(float(row.get("total_dunning", 0)))) + " times\n"
            "- Avg Delay Pay  : " + str(round(float(row.get("avg_payment_delay", 0)), 1)) + " days\n"
            "- Last Login     : " + str(int(float(row["days_since_login"]))) + " days ago\n"
            "- NPS Score      : " + str(round(float(row["avg_nps_score"]), 1)) + "/10\n"
            "- Total Tickets  : " + str(int(float(row.get("total_tickets", 0)))) + "\n"
            "- Avg Severity   : " + str(round(float(row.get("avg_severity", 0)), 2)) + "\n"
            "- Severe Ratio   : " + str(round(float(row.get("severe_ticket_ratio", 0)) * 100, 1)) + "%\n"
            "- Usage          : " + str(round(float(row["avg_usage_hrs"]), 1)) + " hours/month\n"
            "- Feature Adopt  : " + str(round(float(row["avg_feature_adoption"]), 1)) + "%\n"
        )
    return (
        "Data pelanggan:\n"
        "- ID             : " + str(row["customer_id"]) + "\n"
        "- Plan           : " + str(row["plan_type"]) + "\n"
        "- Contract       : " + str(row.get("contract_type", "-")) + "\n"
        "- Tipe Pelanggan : " + str(row.get("customer_type", "-")) + "\n"
        "- Tenure         : " + str(round(float(row["tenure_months"]), 1)) + " bulan\n"
        "- MRR            : Rp " + str(round(float(row.get("mrr", 0)), 2)) + "\n"
        "- Total Payment  : Rp " + str(round(float(row.get("total_payment_value", 0)), 2)) + "\n"
        "- Churn Score    : " + str(round(float(row["churn_score"]), 1)) + "/100\n"
        "- Risk Level     : " + str(row["risk_level"]) + "\n"
        "- Dunning        : " + str(int(float(row.get("total_dunning", 0)))) + " kali\n"
        "- Avg Delay Bayar: " + str(round(float(row.get("avg_payment_delay", 0)), 1)) + " hari\n"
        "- Last Login     : " + str(int(float(row["days_since_login"]))) + " hari lalu\n"
        "- NPS Score      : " + str(round(float(row["avg_nps_score"]), 1)) + "/10\n"
        "- Total Tickets  : " + str(int(float(row.get("total_tickets", 0)))) + "\n"
        "- Avg Severity   : " + str(round(float(row.get("avg_severity", 0)), 2)) + "\n"
        "- Severe Ratio   : " + str(round(float(row.get("severe_ticket_ratio", 0)) * 100, 1)) + "%\n"
        "- Usage          : " + str(round(float(row["avg_usage_hrs"]), 1)) + " jam/bulan\n"
        "- Feature Adopt  : " + str(round(float(row["avg_feature_adoption"]), 1)) + "%\n"
    )


def build_prompt_naratif(row, df: pd.DataFrame = None, lang="id") -> str:
    """
    Chain-of-thought prompt yang kaya konteks:
    - Benchmarking vs. dataset
    - Anomaly detection
    - Risk trajectory
    - Instruksi reasoning bertahap sebelum menulis
    """
    is_en = lang.lower() == "en"
    sinyal = []

    dunning = float(row.get("total_dunning", 0))
    if dunning >= 3:
        sinyal.append(f"KRITIS: {int(dunning)} kali gagal bayar — indikasi kesulitan finansial atau niat churn aktif" if not is_en else f"CRITICAL: {int(dunning)} payment failures — indicator of financial difficulty or active churn intent")
    elif dunning >= 1:
        sinyal.append(f"WASPADA: {int(dunning)} kali gagal bayar" if not is_en else f"WARNING: {int(dunning)} payment failures")

    delay = float(row.get("avg_payment_delay", 0))
    if delay > 10:
        sinyal.append(f"KRITIS: rata-rata keterlambatan bayar {round(delay,1)} hari — sangat tidak normal" if not is_en else f"CRITICAL: average payment delay of {round(delay,1)} days — highly abnormal")
    elif delay > 5:
        sinyal.append(f"WASPADA: rata-rata keterlambatan bayar {round(delay,1)} hari" if not is_en else f"WARNING: average payment delay of {round(delay,1)} days")

    login_days = float(row["days_since_login"])
    if login_days >= 60:
        sinyal.append(f"KRITIS: tidak login selama {int(login_days)} hari — kemungkinan sudah berhenti menggunakan produk" if not is_en else f"CRITICAL: no login for {int(login_days)} days — likely stopped using the product")
    elif login_days >= 30:
        sinyal.append(f"WASPADA: tidak login {int(login_days)} hari — engagement menurun drastis" if not is_en else f"WARNING: no login for {int(login_days)} days — engagement dropped drastically")

    nps = float(row["avg_nps_score"])
    if nps <= 3:
        sinyal.append(f"KRITIS: NPS hanya {round(nps,1)}/10 — pelanggan adalah detractor aktif" if not is_en else f"CRITICAL: NPS is only {round(nps,1)}/10 — active detractor customer")
    elif nps <= 5:
        sinyal.append(f"WASPADA: NPS rendah {round(nps,1)}/10 — pelanggan tidak puas" if not is_en else f"WARNING: low NPS of {round(nps,1)}/10 — unsatisfied customer")
    elif nps >= 9:
        sinyal.append(f"POSITIF: NPS tinggi {round(nps,1)}/10 — pelanggan promoter" if not is_en else f"POSITIVE: high NPS of {round(nps,1)}/10 — promoter customer")

    usage = float(row["avg_usage_hrs"])
    if usage < 5:
        sinyal.append(f"KRITIS: usage sangat rendah {round(usage,1)} jam/bulan — hampir tidak menggunakan produk" if not is_en else f"CRITICAL: very low usage of {round(usage,1)} hours/month — barely using the product")
    elif usage < 15:
        sinyal.append(f"WASPADA: usage rendah {round(usage,1)} jam/bulan" if not is_en else f"WARNING: low usage of {round(usage,1)} hours/month")
    elif usage > 100:
        sinyal.append(f"POSITIF: pengguna sangat aktif {round(usage,1)} jam/bulan" if not is_en else f"POSITIVE: highly active user with {round(usage,1)} hours/month")

    adoption = float(row["avg_feature_adoption"])
    if adoption < 20:
        sinyal.append(f"KRITIS: adopsi fitur hanya {round(adoption,1)}% — pelanggan belum menemukan value produk" if not is_en else f"CRITICAL: feature adoption is only {round(adoption,1)}% — customer hasn't found value yet")
    elif adoption < 40:
        sinyal.append(f"WASPADA: adopsi fitur rendah {round(adoption,1)}%" if not is_en else f"WARNING: low feature adoption of {round(adoption,1)}%")

    ct = str(row.get("customer_type", "")).lower()
    if "problematic" in ct:
        sinyal.append("KRITIS: tipe 'Problematic' — pola tiket komplain berulang dengan severity tinggi, relasi sudah tegang" if not is_en else "CRITICAL: 'Problematic' type — recurring high-severity complaints, strained relationship")
    elif "at_risk" in ct or "at risk" in ct:
        sinyal.append("WASPADA: tipe 'At Risk' — sinyal ketidakpuasan dari pola support ticket" if not is_en else "WARNING: 'At Risk' type — signs of dissatisfaction from support tickets")

    sev_ratio = float(row.get("severe_ticket_ratio", 0))
    if sev_ratio > 0.5:
        sinyal.append(f"KRITIS: {round(sev_ratio*100,0)}% tiket berstatus severity parah — frustrasi terakumulasi" if not is_en else f"CRITICAL: {round(sev_ratio*100,0)}% of tickets have severe severity — accumulated frustration")
    elif sev_ratio > 0.3:
        sinyal.append(f"WASPADA: {round(sev_ratio*100,0)}% tiket berstatus parah" if not is_en else f"WARNING: {round(sev_ratio*100,0)}% of tickets are severe")

    # --- Anomali dari dataset benchmark ---
    anomali_text = ""
    if df is not None:
        anomalies = detect_anomaly(row, df, lang)
        if anomalies:
            anomali_text = "\n\nANOMALI YANG TERDETEKSI (dibanding pelanggan lain):\n" if not is_en else "\n\nDETECTED ANOMALIES (compared to other customers):\n"
            for a in anomalies:
                anomali_text += f"[!] {a}\n"

    # --- Benchmark vs. dataset ---
    benchmark_text = ""
    if df is not None:
        bm = get_dataset_benchmarks(df)
        benchmark_text = ("\n\nPOSISI RELATIF DIBANDING SELURUH PELANGGAN:\n" if not is_en else "\n\nRELATIVE POSITION COMPARED TO ALL CUSTOMERS:\n") + format_benchmarks_text(row, bm, lang)

    # --- Risk level framing ---
    risk = row["risk_level"]
    if risk == "High":
        urgensi = "SANGAT TINGGI (churn bisa terjadi dalam 30 hari)" if not is_en else "VERY HIGH (churn could occur within 30 days)"
        tone    = "Gunakan nada profesional yang menunjukkan urgency. Tindakan harus bisa dilakukan hari ini atau minggu ini." if not is_en else "Use a professional tone showing urgency. Actions must be executable today or this week."
    elif risk == "Medium":
        urgensi = "SEDANG (window 1-2 bulan untuk intervensi)" if not is_en else "MEDIUM (1-2 months window for intervention)"
        tone    = "Gunakan nada proaktif dan preventif. Fokus pada membangun kembali engagement." if not is_en else "Use a proactive and preventive tone. Focus on rebuilding engagement."
    else:
        urgensi = "RENDAH (pelanggan relatif sehat, fokus pada penguatan loyalitas)" if not is_en else "LOW (relatively healthy customer, focus on strengthening loyalty)"
        tone    = "Gunakan nada apresiasi dan proaktif. Identifikasi peluang upsell atau advocacy." if not is_en else "Use an appreciative and proactive tone. Identify upsell or advocacy opportunities."

    sinyal_str = "\n".join(f"• {s}" for s in sinyal) if sinyal else ("• Tidak ada sinyal negatif yang signifikan" if not is_en else "• No significant negative signals")

    if is_en:
        return f"""You will analyze customer {row['customer_id']} and write a retention playbook/recommendation.

REASONING STEPS (think first, do not write output directly):
1. Identify: what are the 2-3 MOST IMPORTANT signals from the data below?
2. Connect: what is the causal relationship between these signals?
3. Predict: if no action is taken, what will happen in 30-60 days?
4. Prioritize: what action is most impactful with minimal resources?
5. After reasoning is complete, write the final output in a 3-paragraph format.

CUSTOMER DATA:
{context_pelanggan(row, lang)}
DETECTED SIGNALS:
{sinyal_str}
CHURN RISK LEVEL: {urgensi}
{benchmark_text}{anomali_text}

OUTPUT INSTRUCTIONS:
Write exactly 3 flowing paragraphs in fluent, professional English. {tone}
- Paragraph 1: Describe this customer's situation concretely and empathetically. Mention the most important numbers.
- Paragraph 2: Explain WHY this is happening — connect behavior patterns, do not just list problems.
- Paragraph 3: Recommend three concrete actions with the format: [WHO] does [WHAT] in [WHEN] because [REASON], with target outcome [EXPECTED RESULT].

Do NOT write any labels (like "Paragraph 1:" or headers). Directly write the paragraph content.
"""

    return f"""Kamu akan menganalisis pelanggan {row['customer_id']} dan menulis rekomendasi retensi.

LANGKAH REASONING (pikirkan dulu, jangan langsung tulis output):
1. Identifikasi: apa 2-3 sinyal PALING PENTING dari data di bawah?
2. Hubungkan: apa hubungan kausalitas antar sinyal tersebut?
3. Prediksi: jika tidak ada tindakan, apa yang akan terjadi dalam 30-60 hari?
4. Prioritaskan: tindakan apa yang paling berdampak dengan resources minimal?
5. Setelah reasoning selesai, tulis output akhir dalam format 3 paragraf.

DATA PELANGGAN:
{context_pelanggan(row, lang)}
SINYAL TERDETEKSI:
{sinyal_str}
TINGKAT RISIKO CHURN: {urgensi}
{benchmark_text}{anomali_text}

INSTRUKSI OUTPUT:
Tulis 3 paragraf mengalir dalam Bahasa Indonesia profesional. {tone}
- Paragraf 1: Gambarkan situasi pelanggan ini secara konkret dan empatik. Sebutkan angka yang paling penting.
- Paragraf 2: Jelaskan MENGAPA ini terjadi — hubungkan pola perilaku, bukan sekadar daftar masalah.
- Paragraf 3: Tiga tindakan konkret dengan format: [SIAPA] melakukan [APA] dalam [KAPAN] karena [ALASAN], dengan target outcome [HASIL YANG DIHARAPKAN].

JANGAN tulis label apapun. Langsung tulis isi paragrafnya."""


def build_chat_context(df: pd.DataFrame, lang="id") -> str:
    """
    Bangun konteks yang kaya untuk chatbot — bukan hanya statistik dasar,
    tapi juga distribusi per segmen, anomali, dan insight menarik.
    """
    n_total  = len(df)
    n_high   = int((df["risk_level"] == "High").sum())
    n_medium = int((df["risk_level"] == "Medium").sum())
    n_low    = int((df["risk_level"] == "Low").sum())
    is_en = lang.lower() == "en"

    if is_en:
        ctx = (
            f"=== ACTUAL DATASET CHURNSHIELD ({n_total} customers) ===\n\n"
            f"RISK DISTRIBUTION:\n"
            f"• High Risk  : {n_high} customers ({round(n_high/n_total*100,1)}%)\n"
            f"• Medium Risk: {n_medium} customers ({round(n_medium/n_total*100,1)}%)\n"
            f"• Low Risk   : {n_low} customers ({round(n_low/n_total*100,1)}%)\n"
            f"• Avg Churn Score: {round(float(df['churn_score'].mean()),1)}/100\n\n"
        )
        if "plan_type" in df.columns:
            ctx += "DISTRIBUTION BY PLAN TYPE:\n"
            for plan, grp in df.groupby("plan_type"):
                h = int((grp["risk_level"] == "High").sum())
                ctx += (
                    f"• {plan}: {len(grp)} customers, {h} High Risk "
                    f"({round(h/len(grp)*100,1)}%), "
                    f"avg NPS: {round(float(grp['avg_nps_score'].mean()),1)}\n"
                )
            ctx += "\n"
        if "customer_type" in df.columns:
            ctx += "DISTRIBUTION BY CUSTOMER TYPE:\n"
            for ct, grp in df.groupby("customer_type"):
                ctx += (
                    f"• {ct}: {len(grp)} customers, "
                    f"avg churn score: {round(float(grp['churn_score'].mean()),1)}, "
                    f"avg NPS: {round(float(grp['avg_nps_score'].mean()),1)}\n"
                )
            ctx += "\n"
        top5 = df[df["risk_level"] == "High"].nlargest(5, "churn_score")
        ctx += "TOP 5 HIGHEST CHURN RISK CUSTOMERS:\n"
        for _, r in top5.iterrows():
            ctx += (
                f"• {r['customer_id']} | {r['plan_type']} | "
                f"Score: {r['churn_score']} | NPS: {round(float(r['avg_nps_score']),1)} | "
                f"Login: {int(float(r['days_since_login']))} days ago | "
                f"Type: {r.get('customer_type', '-')}\n"
            )
        ctx += "\nGLOBAL DATASET ANOMALIES TO NOTE:\n"
        veteran_churn = df[(df["tenure_months"] >= 18) & (df["churn_score"] >= 65)]
        if len(veteran_churn) > 0:
            ctx += f"- {len(veteran_churn)} veteran customers (>=18 months) with churn score >=65 - 'veteran churn' pattern\n"
        trapped = df[(df["avg_usage_hrs"] >= 50) & (df["avg_nps_score"] <= 5)]
        if len(trapped) > 0:
            ctx += f"- {len(trapped)} active customers (>=50 hrs/mo) but NPS <=5 - 'trapped customer' pattern\n"
        dunning_heavy = df[df["total_dunning"] >= 3]
        if len(dunning_heavy) > 0:
            ctx += f"- {len(dunning_heavy)} customers with >=3 dunning - serious payment issue risk\n"
        return ctx

    ctx = (
        f"=== DATA AKTUAL DATASET CHURNSHIELD ({n_total} pelanggan) ===\n\n"
        f"DISTRIBUSI RISIKO:\n"
        f"• High Risk  : {n_high} pelanggan ({round(n_high/n_total*100,1)}%)\n"
        f"• Medium Risk: {n_medium} pelanggan ({round(n_medium/n_total*100,1)}%)\n"
        f"• Low Risk   : {n_low} pelanggan ({round(n_low/n_total*100,1)}%)\n"
        f"• Avg Churn Score: {round(float(df['churn_score'].mean()),1)}/100\n\n"
    )
    if "plan_type" in df.columns:
        ctx += "DISTRIBUSI PER PLAN TYPE:\n"
        for plan, grp in df.groupby("plan_type"):
            h = int((grp["risk_level"] == "High").sum())
            ctx += (
                f"• {plan}: {len(grp)} pelanggan, {h} High Risk "
                f"({round(h/len(grp)*100,1)}%), "
                f"avg NPS: {round(float(grp['avg_nps_score'].mean()),1)}\n"
            )
        ctx += "\n"
    if "customer_type" in df.columns:
        ctx += "DISTRIBUSI PER TIPE PELANGGAN:\n"
        for ct, grp in df.groupby("customer_type"):
            ctx += (
                f"• {ct}: {len(grp)} pelanggan, "
                f"avg churn score: {round(float(grp['churn_score'].mean()),1)}, "
                f"avg NPS: {round(float(grp['avg_nps_score'].mean()),1)}\n"
            )
        ctx += "\n"
    top5 = df[df["risk_level"] == "High"].nlargest(5, "churn_score")
    ctx += "TOP 5 PELANGGAN RISIKO TERTINGGI:\n"
    for _, r in top5.iterrows():
        ctx += (
            f"• {r['customer_id']} | {r['plan_type']} | "
            f"Score: {r['churn_score']} | NPS: {round(float(r['avg_nps_score']),1)} | "
            f"Login: {int(float(r['days_since_login']))} hari lalu | "
            f"Tipe: {r.get('customer_type', '-')}\n"
        )
    ctx += "\nANOMALI DATASET YANG PERLU DIPERHATIKAN:\n"
    veteran_churn = df[(df["tenure_months"] >= 18) & (df["churn_score"] >= 65)]
    if len(veteran_churn) > 0:
        ctx += f"- {len(veteran_churn)} pelanggan veteran (>=18 bln) dengan churn score >=65 - pola 'veteran churn'\n"
    trapped = df[(df["avg_usage_hrs"] >= 50) & (df["avg_nps_score"] <= 5)]
    if len(trapped) > 0:
        ctx += f"- {len(trapped)} pelanggan aktif (>=50 jam/bln) tapi NPS <=5 - 'trapped customer'\n"
    dunning_heavy = df[df["total_dunning"] >= 3]
    if len(dunning_heavy) > 0:
        ctx += f"- {len(dunning_heavy)} pelanggan dengan >=3 kali dunning - risiko payment issue serius\n"
    return ctx


# -------------------------------------------------------
# KATEGORISASI — LLM-based (lebih akurat dari keyword)
# -------------------------------------------------------
KATEGORI_VALID = [
    "Hubungi Langsung",
    "Email/Pesan",
    "Penawaran Diskon",
    "Training/Demo",
    "Eskalasi Support",
    "Renewal Kontrak",
    "Loyalty Program",
    "Upgrade Plan",
    "Churn Recovery",
    "Monitoring",
]

def kategorisasi_llm(rekomendasi: str, row=None) -> str:
    """
    Gunakan LLM untuk klasifikasi kategori tindakan — lebih akurat dari keyword matching.
    Fallback ke keyword matching jika LLM gagal atau timeout.
    """
    if not rekomendasi or rekomendasi.startswith("Error"):
        return _kategorisasi_keyword(rekomendasi)

    kategori_list = ", ".join(f'"{k}"' for k in KATEGORI_VALID)
    prompt = (
        f"Klasifikasikan tindakan utama yang direkomendasikan dalam teks berikut "
        f"ke dalam SATU kategori dari daftar ini:\n{kategori_list}\n\n"
        f"Teks rekomendasi:\n{rekomendasi[:600]}\n\n"
        f"Jawab HANYA dengan nama kategori, tanpa penjelasan apapun."
    )
    messages = [{"role": "user", "content": prompt}]
    result = chat_llm(messages, max_tokens=20)

    # Validasi output LLM
    result_clean = result.strip().strip('"')
    for k in KATEGORI_VALID:
        if k.lower() in result_clean.lower():
            return k

    # Fallback jika LLM tidak return kategori valid
    return _kategorisasi_keyword(rekomendasi)


def _kategorisasi_keyword(rekomendasi: str) -> str:
    """Fallback keyword-based categorization."""
    teks = (rekomendasi or "").lower()
    if any(w in teks for w in ["telepon", "hubungi", "panggil", "call"]):
        return "Hubungi Langsung"
    elif any(w in teks for w in ["email", "kirim pesan", "whatsapp", "wa "]):
        return "Email/Pesan"
    elif any(w in teks for w in ["diskon", "promo", "penawaran", "gratis", "potongan"]):
        return "Penawaran Diskon"
    elif any(w in teks for w in ["training", "onboarding", "demo", "tutorial", "webinar"]):
        return "Training/Demo"
    elif any(w in teks for w in ["tiket", "support", "selesaikan", "eskalasi", "bug", "masalah teknis"]):
        return "Eskalasi Support"
    elif any(w in teks for w in ["kontrak", "perpanjang", "renewal"]):
        return "Renewal Kontrak"
    elif any(w in teks for w in ["reward", "loyalitas", "apresiasi", "loyalty"]):
        return "Loyalty Program"
    elif any(w in teks for w in ["upgrade", "paket lebih", "plan lebih"]):
        return "Upgrade Plan"
    elif any(w in teks for w in ["recovery", "darurat", "segera selamatkan"]):
        return "Churn Recovery"
    else:
        return "Monitoring"


# Alias untuk backward compatibility
def kategorisasi(rekomendasi: str) -> str:
    return kategorisasi_llm(rekomendasi)


# -------------------------------------------------------
# INTENT DETECTION untuk CHAT
# -------------------------------------------------------
def detect_chat_intent(message: str) -> str:
    """
    Klasifikasi intent pesan user untuk memilih strategi respons yang tepat.
    Return: 'faktual' | 'analitis' | 'strategis' | 'perbandingan'
    """
    msg = message.lower()
    if any(w in msg for w in ["siapa", "berapa", "mana", "id", "nama", "list", "daftar", "tampilkan", "cari"]):
        return "faktual"
    elif any(w in msg for w in ["kenapa", "mengapa", "penyebab", "alasan", "bagaimana bisa", "pola", "tren"]):
        return "analitis"
    elif any(w in msg for w in ["strategi", "saran", "rekomendasi", "apa yang", "harus", "langkah", "plan", "tindakan"]):
        return "strategis"
    elif any(w in msg for w in ["banding", "dibanding", "vs", "versus", "lebih", "antara", "segmen", "tipe"]):
        return "perbandingan"
    return "analitis"  # default ke analitis


# -------------------------------------------------------
# PYDANTIC MODELS
# -------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    session_id: Optional[str] = None   # untuk conversation history
    lang: Optional[str] = "id"         # "id" atau "en"

class ChatResponse(BaseModel):
    reply: str
    customer_id: Optional[str] = None
    session_id: str
    intent: Optional[str] = None


# -------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "selamat datang di churnshield api - visions project!",
        "status": "aktif",
        "docs_url": "/docs"
    }


@app.get("/health")
def health():
    return {
        "groq_api":        "ok" if test_groq() else "down — cek GROQ_API_KEY",
        "ml_model":        "loaded" if model is not None else "not found",
        "data_loaded":     _df is not None,
        "total_pelanggan": len(_df) if _df is not None else 0,
    }


@app.get("/statistik")
def statistik():
    df = get_df()
    return {
        "total":           len(df),
        "high_risk":       int((df["risk_level"] == "High").sum()),
        "medium_risk":     int((df["risk_level"] == "Medium").sum()),
        "low_risk":        int((df["risk_level"] == "Low").sum()),
        "avg_churn_score": round(float(df["churn_score"].mean()), 1),
        "avg_nps":         round(float(df["avg_nps_score"].mean()), 1),
    }


@app.post("/reload")
def reload_data():
    global _df
    _df = load_data()
    return {"status": "ok", "total_pelanggan": len(_df) if _df is not None else 0}


@app.get("/pelanggan")
def list_pelanggan(risk: Optional[str] = None, limit: int = 20):
    df = get_df()
    if risk:
        df = df[df["risk_level"] == risk.capitalize()]
    return df.sort_values("churn_score", ascending=False).head(limit).to_dict(orient="records")


@app.get("/pelanggan/{customer_id}")
def get_pelanggan(customer_id: str):
    df = get_df()
    result = df[df["customer_id"].str.upper() == customer_id.upper()]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Pelanggan '{customer_id}' tidak ditemukan")
    return result.iloc[0].to_dict()


@app.post("/rekomendasi/{customer_id}")
def get_rekomendasi(customer_id: str, lang: str = "id"):
    df = get_df()
    result = df[df["customer_id"].str.upper() == customer_id.upper()]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Pelanggan '{customer_id}' tidak ditemukan")
    row = result.iloc[0]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EN if lang.lower() == "en" else SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt_naratif(row, df, lang)},
    ]

    rekomendasi = chat_llm(messages, max_tokens=800)
    prioritas   = "URGENT" if row["risk_level"] == "High" else "NORMAL" if row["risk_level"] == "Medium" else "LOW"

    return {
        "customer_id":       row["customer_id"],
        "risk_level":        row["risk_level"],
        "churn_score":       round(float(row["churn_score"]), 1),
        "kategori_tindakan": kategorisasi_llm(rekomendasi, row),
        "prioritas":         prioritas,
        "rekomendasi":       rekomendasi,
    }


@app.post("/analisis/{customer_id}")
def get_analisis(customer_id: str, lang: str = "id"):
    """
    Analisis mendalam: root cause hypotheses + prediksi timeline + action scoring.
    Menggunakan model yang lebih powerful (70b) untuk kualitas output lebih tinggi.
    """
    df = get_df()
    result = df[df["customer_id"].str.upper() == customer_id.upper()]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Pelanggan '{customer_id}' tidak ditemukan")
    row = result.iloc[0]
    is_en = lang.lower() == "en"

    # Deteksi anomali dan benchmark
    anomalies = detect_anomaly(row, df, lang)
    bm        = get_dataset_benchmarks(df)
    bench_txt = format_benchmarks_text(row, bm, lang)

    # Hitung percentile churn score pelanggan ini
    churn_pct = int(round(float((df["churn_score"] <= row["churn_score"]).mean() * 100), 0))

    if is_en:
        analisis_prompt = f"""You are a senior Customer Success Analyst asked to perform a deep diagnosis.

CUSTOMER DATA:
{context_pelanggan(row, lang)}

RELATIVE POSITION (vs. all {len(df)} customers):
{bench_txt}
• Churn Score Percentile: {churn_pct}th percentile (more at risk than {churn_pct}% of other customers)

DETECTED ANOMALIES:
{chr(10).join('⚠️  ' + a for a in anomalies) if anomalies else '• No significant anomalies'}

ANALYSIS TASK (write in fluent, professional English, without headers or bullet points):

Section 1 — SITUATION & CRITICAL FACTS (1 paragraph):
Describe this customer's condition comprehensively. Which numbers are most concerning and why? Contextualize with their relative position in the dataset.

Section 2 — ROOT CAUSE HYPOTHESES (1 paragraph):
Identify 2-3 main possible causes that could explain this pattern. Use causal reasoning: "X is likely caused by Y because of Z." Separate what can be confirmed from the data and what remains a hypothesis.

Section 3 — PREDICTION WITHOUT INTERVENTION (1 paragraph):
If no action is taken within 30 days, 60 days, and 90 days — what is likely to happen? Mention the concrete revenue risk if this customer's MRR is lost.

Section 4 — PRIORITIZED ACTION PLAN (1 paragraph):
Recommend 3 sequential actions starting from the highest urgency. For each action, state: who performs it, in how many days, what metric is the target, and its estimated impact on churn probability.

IMPORTANT: Write as flowing paragraphs. Do NOT use headers, bullet points, or labels like "Section 1". Write the content directly."""
    else:
        analisis_prompt = f"""Kamu adalah Customer Success Analyst senior yang diminta melakukan diagnosa mendalam.

DATA PELANGGAN:
{context_pelanggan(row, lang)}

POSISI RELATIF (vs. seluruh {len(df)} pelanggan):
{bench_txt}
• Churn Score Percentile: {churn_pct}th percentile (lebih berisiko dari {churn_pct}% pelanggan lain)

ANOMALI TERDETEKSI:
{chr(10).join('⚠️  ' + a for a in anomalies) if anomalies else '• Tidak ada anomali signifikan'}

TUGAS ANALISIS (tulis dalam Bahasa Indonesia profesional, tanpa header atau bullet point):

Bagian 1 — SITUASI & FAKTA KRITIS (1 paragraf):
Deskripsikan kondisi pelanggan ini secara komprehensif. Angka mana yang paling mengkhawatirkan dan mengapa? Kontekstualisasikan dengan posisi relatifnya di dataset.

Bagian 2 — ROOT CAUSE HYPOTHESES (1 paragraf):
Identifikasi 2-3 kemungkinan penyebab utama yang bisa menjelaskan pola ini. Gunakan reasoning kausal: "X kemungkinan disebabkan oleh Y karena Z." Pisahkan mana yang bisa dikonfirmasi dari data dan mana yang masih hipotesis.

Bagian 3 — PREDIKSI TANPA INTERVENSI (1 paragraf):
Jika tidak ada tindakan dalam 30 hari, 60 hari, dan 90 hari — apa yang kemungkinan terjadi? Sebutkan revenue risk yang konkret jika MRR pelanggan ini hilang.

Bagian 4 — ACTION PLAN PRIORITAS (1 paragraf):
Rekonstruksi 3 tindakan berurutan dari urgency tertinggi. Untuk setiap tindakan sebutkan: siapa yang melakukan, dalam berapa hari, metrik apa yang jadi target, dan estimasi impact-nya terhadap churn probability.

PENTING: Tulis mengalir sebagai paragraf, JANGAN gunakan header, bullet point, atau label seperti "Bagian 1". Langsung tulis kontennya."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EN if is_en else SYSTEM_PROMPT},
        {"role": "user",   "content": analisis_prompt},
    ]

    # Coba model 70b dulu, fallback ke 8b jika gagal
    analisis = chat_llm(messages, max_tokens=1000, model=GROQ_MODEL_DEEP)
    if analisis.startswith("Error"):
        analisis = chat_llm(messages, max_tokens=1000, model=GROQ_MODEL)

    return {
        "customer_id":       row["customer_id"],
        "risk_level":        row["risk_level"],
        "churn_score":       round(float(row["churn_score"]), 1),
        "churn_percentile":  churn_pct,
        "anomali_terdeteksi": anomalies,
        "analisis":          analisis,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chatbot dengan conversation history, intent detection, dan enriched context.
    """
    df = get_df()
    is_en = req.lang.lower() == "en"

    # --- Session management ---
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []

    history = _chat_sessions[session_id]

    # --- Detect intent ---
    intent = detect_chat_intent(req.message)

    # --- Bangun messages dengan system prompts ---
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT_EN if is_en else CHAT_SYSTEM_PROMPT}]

    # Inject enriched context dataset
    ctx = build_chat_context(df, req.lang)
    messages.append({"role": "system", "content": ctx})

    # Intent-specific instruction
    intent_instruction_id = {
        "faktual":       "Jawab LANGSUNG dan SINGKAT dengan data konkret. Sebutkan Customer ID spesifik. Gunakan angka.",
        "analitis":      "Berikan analisis mendalam dengan reasoning kausal. Hubungkan pola antar metrik. Identifikasi insight yang tidak obvious.",
        "strategis":     "Berikan framework tindakan yang terstruktur: prioritas, timeline, dan expected outcome. Spesifik dan actionable.",
        "perbandingan":  "Bandingkan segmen/pelanggan dengan data konkret. Gunakan angka dan persentase. Highlight perbedaan signifikan.",
    }
    intent_instruction_en = {
        "faktual":       "Answer DIRECTLY and BRIEFLY with concrete data. Mention specific Customer ID. Use numbers.",
        "analitis":      "Provide deep analysis with causal reasoning. Connect patterns between metrics. Identify non-obvious insights.",
        "strategis":     "Provide a structured framework of actions: priorities, timeline, and expected outcomes. Specific and actionable.",
        "perbandingan":  "Compare segments/customers with concrete data. Use numbers and percentages. Highlight significant differences.",
    }
    intent_instruction = intent_instruction_en if is_en else intent_instruction_id
    messages.append({"role": "system", "content": f"MODE RESPONS: {intent_instruction.get(intent, '')}"})

    # Inject language preference instruction
    if is_en:
        messages.append({"role": "system", "content": "IMPORTANT: Respond in fluent, professional English. All analyses, summaries, and recommendations must be written in English."})
    else:
        messages.append({"role": "system", "content": "PENTING: Jawab dalam Bahasa Indonesia yang profesional dan conversational."})

    # Inject detail pelanggan jika ada customer_id
    if req.customer_id:
        result = df[df["customer_id"].str.upper() == req.customer_id.upper()]
        if result.empty:
            raise HTTPException(status_code=404, detail=f"Pelanggan '{req.customer_id}' tidak ditemukan")
        row = result.iloc[0]
        
        detail_header = "CUSTOMER DETAIL CURRENTLY UNDER DISCUSSION:\n" if is_en else "DETAIL PELANGGAN YANG SEDANG DIBAHAS:\n"
        detail = detail_header + context_pelanggan(row, req.lang)

        # Tambahkan anomali pelanggan ini ke context
        anomalies = detect_anomaly(row, df, req.lang)
        if anomalies:
            anomali_header = "\nDETECTED ANOMALIES:\n" if is_en else "\nANOMALI YANG TERDETEKSI:\n"
            detail += anomali_header + "\n".join(f"[!] {a}" for a in anomalies)

        if "rekomendasi_llm" in row and pd.notna(row["rekomendasi_llm"]) and str(row["rekomendasi_llm"]) != "0":
            rekomendasi_header = "\nEXISTING RECOMMENDATION PLAYBOOK:\n" if is_en else "\nREKOMENDASI YANG SUDAH ADA:\n"
            detail += rekomendasi_header + str(row['rekomendasi_llm'])
        messages.append({"role": "system", "content": detail})

    # Inject conversation history (max N pesan terakhir)
    for h in history[-CHAT_MAX_HISTORY:]:
        messages.append(h)

    # Pesan user saat ini
    messages.append({"role": "user", "content": req.message})

    # --- Panggil LLM ---
    reply = chat_llm(messages, max_tokens=700)

    # --- Simpan ke history ---
    history.append({"role": "user",      "content": req.message})
    history.append({"role": "assistant", "content": reply})
    # Trim history
    if len(history) > CHAT_MAX_HISTORY * 2:
        _chat_sessions[session_id] = history[-(CHAT_MAX_HISTORY * 2):]

    return ChatResponse(reply=reply, customer_id=req.customer_id, session_id=session_id, intent=intent)


@app.delete("/chat/session/{session_id}")
def clear_chat_session(session_id: str):
    """Reset conversation history untuk sesi tertentu."""
    _chat_sessions.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


@app.get("/chat/session/{session_id}")
def get_chat_history(session_id: str):
    """Ambil riwayat percakapan untuk sesi tertentu."""
    if session_id not in _chat_sessions:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    return {"session_id": session_id, "history": _chat_sessions[session_id]}


# -------------------------------------------------------
# RETENSI OTOMATIS (batch LLM processing)
# -------------------------------------------------------
def _run_retensi(df_sorted: pd.DataFrame):
    global _retensi_progress
    total = len(df_sorted)
    _retensi_progress = {"running": True, "done": 0, "total": total, "status": "running"}
    rekomendasi_list, kategori_list, prioritas_list = [], [], []

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt_naratif(row, _df)},
        ]
        rek = chat_llm_batch(messages)
        rekomendasi_list.append(rek)
        kategori_list.append(_kategorisasi_keyword(rek))   # pakai keyword fallback untuk kecepatan batch
        prioritas_list.append(
            "URGENT" if row["risk_level"] == "High" else
            "NORMAL" if row["risk_level"] == "Medium" else "LOW"
        )
        _retensi_progress["done"] = i + 1
        time.sleep(2)

        if (i + 1) % 50 == 0:
            df_cp = df_sorted.iloc[: i + 1].copy()
            df_cp["rekomendasi_llm"]   = rekomendasi_list
            df_cp["kategori_tindakan"] = kategori_list
            df_cp["prioritas_llm"]     = prioritas_list
            df_cp.to_csv(OUTPUT_CSV, index=False)

    df_out = df_sorted.copy()
    df_out["rekomendasi_llm"]   = rekomendasi_list
    df_out["kategori_tindakan"] = kategori_list
    df_out["prioritas_llm"]     = prioritas_list
    df_out.to_csv(OUTPUT_CSV, index=False)
    _retensi_progress.update({"running": False, "status": "done"})
    _buat_notifikasi(df_out[df_out["risk_level"] == "High"])


@app.post("/retensi-otomatis/start")
def start_retensi():
    if _retensi_progress["running"]:
        raise HTTPException(status_code=409, detail="Retensi otomatis sedang berjalan")
    df = get_df()
    df_sorted = pd.concat([
        df[df["risk_level"] == "High"].sort_values("churn_score", ascending=False),
        df[df["risk_level"] == "Medium"].sort_values("churn_score", ascending=False),
        df[df["risk_level"] == "Low"].sort_values("churn_score", ascending=False),
    ]).reset_index(drop=True)
    threading.Thread(target=_run_retensi, args=(df_sorted,), daemon=True).start()
    return {"status": "started", "total": len(df_sorted)}


@app.get("/retensi-otomatis/status")
def status_retensi():
    p = _retensi_progress
    return {**p, "percent": round(p["done"] / p["total"] * 100, 1) if p["total"] > 0 else 0}


@app.get("/retensi-otomatis/hasil")
def hasil_retensi(limit: int = 50, risk: Optional[str] = None):
    if not os.path.exists(OUTPUT_CSV):
        raise HTTPException(status_code=404, detail="Belum ada hasil. Jalankan /retensi-otomatis/start dulu.")
    df = pd.read_csv(OUTPUT_CSV)
    if risk:
        df = df[df["risk_level"] == risk.capitalize()]
    return df.head(limit).to_dict(orient="records")


# -------------------------------------------------------
# NOTIFIKASI
# -------------------------------------------------------
@app.get("/notifikasi")
def get_notifikasi(hanya_belum_baca: bool = True, limit: int = 50):
    cfg = get_supabase()
    if cfg is None:
        raise HTTPException(status_code=503, detail="Supabase belum dikonfigurasi")
    params = {"select": "*", "order": "created_at.desc", "limit": str(limit)}
    if hanya_belum_baca:
        params["dibaca"] = "eq.false"
    r = requests.get(cfg["url"] + "/rest/v1/notifikasi", headers=_sb_headers(cfg), params=params, timeout=15)
    return r.json()


@app.patch("/notifikasi/{notif_id}/baca")
def tandai_dibaca(notif_id: str):
    cfg = get_supabase()
    if cfg is None:
        raise HTTPException(status_code=503, detail="Supabase belum dikonfigurasi")
    requests.patch(
        cfg["url"] + "/rest/v1/notifikasi",
        headers={**_sb_headers(cfg), "Prefer": "return=minimal"},
        params={"id": "eq." + notif_id},
        json={"dibaca": True},
        timeout=15,
    )
    return {"status": "ok"}


@app.patch("/notifikasi/baca-semua")
def baca_semua_notifikasi():
    cfg = get_supabase()
    if cfg is None:
        raise HTTPException(status_code=503, detail="Supabase belum dikonfigurasi")
    requests.patch(
        cfg["url"] + "/rest/v1/notifikasi",
        headers={**_sb_headers(cfg), "Prefer": "return=minimal"},
        params={"dibaca": "eq.false"},
        json={"dibaca": True},
        timeout=15,
    )
    return {"status": "ok"}


@app.post("/notifikasi/generate")
def generate_notifikasi():
    df = get_df()
    df_high = df[df["risk_level"] == "High"]
    if df_high.empty:
        return {"status": "ok", "dibuat": 0}
    _buat_notifikasi(df_high)
    return {"status": "ok", "dibuat": len(df_high)}


if __name__ == "__main__":
    import uvicorn
    
    # mengambil port yang dialokasikan oleh pterodactyl, default ke 8000 jika tidak ada
    port = int(os.environ.get("SERVER_PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)