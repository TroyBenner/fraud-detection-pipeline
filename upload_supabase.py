
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_BUCKET = "DTSC_project"
SUPABASE_PATH = "csv/"

CSV_FILENAME = "articles-fraud.csv"  # or dynamic


def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing Supabase credentials")
    return create_client(url, key)


supabase = get_client()


def upload_csv():
    if not os.path.exists(CSV_FILENAME):
        raise FileNotFoundError(f"{CSV_FILENAME} does not exist.")

    supabase_path = SUPABASE_PATH + CSV_FILENAME

    with open(CSV_FILENAME, "rb") as f:
        try:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                supabase_path,
                f,
                file_options={"content-type": "text/csv", "upsert": "true"}
            )
            print("✔ Uploaded to Supabase:", supabase_path)
        except Exception as e:
            print("Upload error:", e)


if __name__ == "__main__":
    upload_csv()
