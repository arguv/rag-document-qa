
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_database_url


def create_database_engine() -> Engine:
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=10,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=30000"},
    )
