from pathlib import Path
from polymer_gc.sec import SimSEC
import json
from model import RandomArchitectureDataset
from tqdm import tqdm


def make_sec(
    mn,
    mw,
):
    max_sec_tries = 50
    tol = 0.005
    rs = 0
    while True:
        try:
            for i in range(max_sec_tries):
                rs += 1
                try:
                    sec = SimSEC.from_mn_mw(
                        mn,
                        mw,
                        random_state=rs,
                        tol=tol,
                        n_points=500,
                        n_points_detect=1000,
                    )
                    return sec
                except ValueError:
                    if i >= max_sec_tries - 1:
                        raise
        except ValueError:
            if tol > 0.05:
                raise
            tol += 0.005


with open("RandomArchitecture.json", "r") as f:
    pg_dataset = RandomArchitectureDataset(**json.load(f))

try:
    for item in tqdm(pg_dataset.items, total=len(pg_dataset.items)):
        if item.sec is None:
            try:
                item.sec_raw = json.loads(
                    make_sec(
                        mn=item.mn,
                        mw=item.mw,
                    )._raw_data.to_json(orient="split")
                )

            except ValueError as e:
                print(
                    f"Failed to create SEC for {item.sequence} with mn={item.mn}, mw={item.mw}: {e}"
                )
                item.sec = None
finally:
    json_path = Path("RandomArchitecture.json")
    json_path.write_text(pg_dataset.model_dump_json(indent=2))
