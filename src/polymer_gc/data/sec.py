from sqlmodel import Column, JSON, Field, Integer, Relationship
from .basemodels.core import Base
import pandas as pd
from ..sec import SimSEC
from .database import SessionRegistry
from typing import Optional
import numpy as np


class SECEntry(Base, table=True):
    id: int = Field(sa_column=Column(Integer, primary_key=True))
    sec_raw: dict = Field(sa_column=Column(JSON))
    sec_calibration_params: list[float] = Field(sa_column=Column(JSON))
    sim_params: dict = Field(default_factory=dict, sa_column=Column(JSON))

    @property
    def sec(self):
        return SimSEC(
            pd.DataFrame(
                self.sec_raw["data"],
                columns=self.sec_raw["columns"],
                index=self.sec_raw["index"],
            ),
            calibration_params=self.sec_calibration_params,
        )

    @sec.setter
    def sec(self, value: SimSEC):
        self.sec_raw = value._raw_data.to_dict(orient="split")
        self.sec_calibration_params = value.calibration_params

    @classmethod
    def from_sec(cls, sec: SimSEC, commit: bool = True, **kwargs):
        session = SessionRegistry.get_session()
        kwargs["sec_raw"] = sec._raw_data.to_dict(orient="split")
        kwargs["sec_calibration_params"] = sec.calibration_params
        new = cls(**kwargs)
        session.add(new)
        if commit:
            session.commit()
            session.refresh(new)
        return new

    def sample_masses(
        self,
        n: int = 1,
        random_state: Optional[int | np.random.Generator] = None,
        by: str = "Mw",
    ):
        return self.sec.sample(n=n, by=by, random_state=random_state)
