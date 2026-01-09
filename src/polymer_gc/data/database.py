import sqlalchemy as sa

from sqlalchemy.orm import DeclarativeBase, relationship, validates
from typing import Dict, Any, Optional, List
from rdkit import Chem
from rdkit.Chem import Descriptors

from sqlmodel import (
    Field,
    SQLModel,
    create_engine,
    Session,
    select,
    JSON,
    Column,
    String,
    Relationship,
)


class SessionRegistry:
    _current_session: Optional[Session] = None

    @classmethod
    def set_session(cls, session: Session):
        cls._current_session = session

    @classmethod
    def get_session(cls) -> Session:
        if cls._current_session is None:
            raise RuntimeError(
                "No active session. Use sessioncontext() to create a session."
            )
        return cls._current_session

    @classmethod
    def clear_session(cls):
        cls._current_session = None


class SessionManager:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.engine = None
        self.session = None
        self.previous_session = None

    def new_engine(self):
        engine = create_engine(f"sqlite:///{self.database_path}")
        SQLModel.metadata.create_all(engine)
        return engine

    def __enter__(self):
        self.engine = self.new_engine()
        self.session = Session(self.engine, expire_on_commit=False)
        # Store the previous session before setting the new one
        try:
            self.previous_session = SessionRegistry.get_session()
        except RuntimeError:
            self.previous_session = None
        SessionRegistry.set_session(self.session)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        # Restore the previous session if it existed
        if self.previous_session is not None:
            SessionRegistry.set_session(self.previous_session)
        else:
            SessionRegistry.clear_session()
