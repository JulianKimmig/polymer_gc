from typing import Dict, Any, List
from sqlmodel import SQLModel, select
from ..database import SessionRegistry


class Base(SQLModel):
    def update(self, **kwargs):
        session = SessionRegistry.get_session()
        changed = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                if old_value != value:
                    setattr(self, key, value)
                    changed = True
        if changed:
            session.commit()
            session.refresh(self)
        return self

    @classmethod
    def all(cls, **filters):
        session = SessionRegistry.get_session()
        statement = select(cls).where(
            *[getattr(cls, key) == value for key, value in filters.items()]
        )
        return session.exec(statement).all()

    @classmethod
    def get(cls, set_kwargs: Dict[str, Any] = None, **filters):
        session = SessionRegistry.get_session()
        statement = select(cls).where(
            *[getattr(cls, key) == value for key, value in filters.items()]
        )

        instance = session.exec(statement).first()
        if not instance:
            return None
        if set_kwargs:
            instance = instance.update(**set_kwargs)

        return instance

    @classmethod
    def fill_values(cls, **kwargs):
        return kwargs

    @classmethod
    def select(cls):
        return select(cls)

    @classmethod
    def get_or_create(
        cls,
        new_kwargs: Dict[str, Any] = None,
        set_kwargs: Dict[str, Any] = None,
        commit: bool = True,
        **filters,
    ):
        """
        Get or create an instance of the model.

        Args:
            new_kwargs: Keyword arguments to be passed to the constructor.
            set_kwargs: Keyword arguments to be set on the instance.
            commit: Whether to commit the session after creating the instance.
            **filters: Filters to be applied to the query.  Type of filters is Dict[str, Any]
        Returns:
        """
        session = SessionRegistry.get_session()
        set_kwargs = set_kwargs or {}
        new_kwargs = new_kwargs or {}
        # try to retrieve the instance
        instance = cls.get(set_kwargs=set_kwargs, **filters)

        if instance:
            return instance

        # Only include non-None values in the constructor kwargs
        all_kwargs = {**filters, **new_kwargs, **set_kwargs}
        # Remove None values to allow validation to set defaults
        constructor_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}

        constructor_kwargs = cls.fill_values(**constructor_kwargs)

        # Create instance with validation
        # print(constructor_kwargs)
        instance = cls(**constructor_kwargs)
        session.add(instance)
        if commit:
            session.commit()
            session.refresh(instance)
        return instance
