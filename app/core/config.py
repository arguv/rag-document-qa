
import os

from dotenv import load_dotenv


load_dotenv()


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required"
        )

    return url
