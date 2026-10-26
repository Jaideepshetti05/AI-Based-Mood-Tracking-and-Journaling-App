from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Lightweight SQLite column migration for analyses table
    if settings.database_url.startswith("sqlite"):
        with engine.connect() as conn:
            columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(analyses)").fetchall()]
            new_columns = [
                ("original_emotion", "VARCHAR(64)"),
                ("original_confidence", "FLOAT"),
                ("personalized_emotion", "VARCHAR(64)"),
                ("personalized_confidence", "FLOAT"),
                ("personalization_applied", "BOOLEAN DEFAULT 0"),
                ("personalization_explanation", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in columns:
                    conn.exec_driver_sql(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}")
            conn.commit()
