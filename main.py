import os, io, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from scipy.stats import rankdata # DITAMBAHKAN UNTUK RANKING SCORE
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- 1. inisialisasi app & cors ---
app = FastAPI(title="ChurnShield API - Visions Project", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. load semua alat tempur ---
try:
    model = joblib.load('model_churn_final.pkl')
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("✅ sukses load semua model pkl!")
except Exception as e:
    print(f"❌ gagal load pkl: {e}")

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

# --- 5. endpoint utama ---
@app.post("/predict-batch")
async def predict_batch(
    file_accounts: UploadFile = File(...),
    file_usage: UploadFile = File(...),
    file_billing: UploadFile = File(...),
    file_tickets: UploadFile = File(...),
    file_nps: UploadFile = File(...)
):
    try:
        # baca semua file mentah
        df_accounts = pd.read_csv(io.BytesIO(await file_accounts.read()))
        df_usage = pd.read_csv(io.BytesIO(await file_usage.read()))
        df_billing = pd.read_csv(io.BytesIO(await file_billing.read()))
        df_tickets = pd.read_csv(io.BytesIO(await file_tickets.read()))
        df_nps = pd.read_csv(io.BytesIO(await file_nps.read()))

        # FIX: contract_type dibiarin kapital kayak aslinya di dataset
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

        # hitung tanggal persis kayak EDA
        df_master['subscription_date'] = pd.to_datetime(df_master['subscription_date'], format='mixed', dayfirst=True, errors='coerce')
        df_master['unsubscribed_date'] = pd.to_datetime(df_master['unsubscribed_date'], format='mixed', dayfirst=True, errors='coerce')
        df_master['last_login'] = pd.to_datetime(df_master['last_login'], format='mixed', dayfirst=True, errors='coerce')
        
        tanggal_acuan = pd.to_datetime('2024-12-31')
        df_master['end_date'] = df_master['unsubscribed_date'].fillna(tanggal_acuan)
        
        df_master['tenure_months'] = ((df_master['end_date'] - df_master['subscription_date']).dt.days / 30).round(1)
        df_master['days_since_login'] = (df_master['end_date'] - df_master['last_login']).dt.days
        df_master['usage_per_user'] = df_master['avg_usage_hrs'] / df_master['total_users'].replace(0,1)

        # =======================================================
        # PROSES FILLNA (IMPUTASI)
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
        
        # FIX: ubah boolean jadi integer
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
        # EKSEKUSI PREDIKSI AKHIR MENGGUNAKAN RANKING (SAMA SEPERTI JUPYTER)
        # =======================================================
        # 1. Ambil probabilitas mentah (index 1)
        probs = model.predict_proba(df_scaled)[:, 1]
        
        # 2. Hitung persentil berbasis ranking
        ranks = rankdata(probs)
        if len(ranks) > 1:
            churn_score_ranked = ((ranks - 1) / (len(ranks) - 1) * 100).round(1)
        else:
            churn_score_ranked = pd.Series([0.0]) # Handle jika datanya cuma 1 baris
            
        df_master['churn_score'] = churn_score_ranked
        
        # 3. Tentukan level risiko
        df_master['risk_level'] = df_master['churn_score'].apply(
            lambda x: 'High' if x >= 70 else ('Medium' if x >= 30 else 'Low')
        )

        # ngerapihin kapital data buat dibalikin ke frontend
        df_master['plan_type'] = df_master['plan_type'].astype(str).str.title()
        df_master['contract_type'] = df_master['contract_type'].astype(str).str.title()
        df_master['customer_type'] = df_master['customer_type'].astype(str).str.replace('_', ' ').str.title()
        df_master['mrr'] = (df_master['total_payment_value'] / df_master['tenure_months'].replace(0,1)).round(2)
        
        # --- DAFTAR LENGKAP KOLOM TERMASUK CHURN & SCORE ---
        output_cols = [
            'customer_id', 'plan_type', 'contract_type', 'customer_type', 'tenure_months', 
            'total_payment_value', 'mrr', 'total_dunning', 'avg_payment_delay', 'days_since_login', 
            'avg_nps_score', 'total_tickets', 'avg_severity', 'severe_ticket_ratio', 
            'avg_usage_hrs', 'usage_per_user', 'avg_feature_adoption', 
            'churn', 'churn_score', 'risk_level'
        ]
        valid_cols = [c for c in output_cols if c in df_master.columns]
        
        output = io.StringIO()
        df_master[valid_cols].to_csv(output, index=False)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=churn_prediction_results.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)