# load_data_supabase.py
import pandas as pd
import io
from supabase_client import supabase, SUPABASE_BUCKET, SUPABASE_CSV_PATH

def load_fraud_data():
    """Loads FINRA fraud CSV stored in Supabase bucket."""
    res = supabase.storage.from_(SUPABASE_BUCKET).download(SUPABASE_CSV_PATH)

    if res is None:
        raise ValueError("Failed to download CSV from Supabase.")

    df = pd.read_csv(io.BytesIO(res))

    # clean + ensure consistent formats
    df["summary"] = df["summary"].astype(str)
    df["keywords"] = df["keywords"].astype(str)

    # turn keywords into lists
    df["keywords"] = df["keywords"].fillna("").apply(
        lambda x: [k.strip().lower() for k in x.split(",") if k.strip()]
    )

    return df
