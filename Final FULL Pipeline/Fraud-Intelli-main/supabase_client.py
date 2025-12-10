# supabase_client.py
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_BUCKET = "DTSC_project"
SUPABASE_CSV_PATH = "csv/articles-fraud.csv"  # exact path from your working file

def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Missing Supabase credentials.")

    return create_client(url, key)

supabase = get_supabase()
