import numpy as np
import pandas as pd
from pathlib import Path

base = Path("data_cache")
base.mkdir(exist_ok=True)

# 1) energy.csv
def make_energy_csv(n=500):
    rng = np.random.default_rng(0)
    built_age = rng.choice(["Pre-30s", "Post-30s"], size=n, p=[0.3, 0.7])
    df = pd.DataFrame({
        "built_age": built_age,
        "current_energy_efficiency": rng.normal(60, 10, size=n).round(1),
        "environment_impact_current": rng.normal(60, 10, size=n).round(1),
        "energy_consumption_current": rng.normal(200, 50, size=n).round(1),
        "co2_emissions_current": rng.normal(4, 1, size=n).round(2),
        "lighting_cost_current": rng.normal(100, 20, size=n).round(2),
        "heating_cost_current": rng.normal(500, 100, size=n).round(2),
        "hot_water_cost_current": rng.normal(150, 30, size=n).round(2),
    })
    df.to_csv(base / "energy.csv", index=False)
    print("Wrote", base / "energy.csv")

# 2) la_energy.csv
def make_la_energy_csv(n_areas=20):
    rng = np.random.default_rng(1)
    records = []
    for i in range(n_areas):
        code = f"E08{rng.integers(0,99999):05d}"
        for age in ["Old", "Recent"]:
            for n_rooms in range(1, 7):
                shortfall = float(rng.normal(4, 1))
                n = int(rng.integers(10, 200))
                records.append({
                    "shortfall": round(shortfall, 2),
                    "local_authority_code": code,
                    "age": age,
                    "n_rooms": n_rooms,
                    "n": n,
                })
    df = pd.DataFrame(records)
    df.to_csv(base / "la_energy.csv", index=False)
    print("Wrote", base / "la_energy.csv")

# 3) la_collision.csv
def make_la_collision_csv(n_areas=30):
    rng = np.random.default_rng(2)
    records = []
    for i in range(n_areas):
        code = f"E06{rng.integers(0,99999):05d}"
        n_total = int(rng.integers(50, 1000))
        # 随便按比例拆分一下各种情况
        n_minis = int(n_total * rng.uniform(0.01, 0.1))
        n_rain = int(n_total * rng.uniform(0.1, 0.4))
        n_dark = int(n_total * rng.uniform(0.1, 0.4))
        n_dry = int(n_total * rng.uniform(0.4, 0.9))
        n_urban = int(n_total * rng.uniform(0.3, 0.9))
        n_slight = int(n_total * rng.uniform(0.5, 0.95))

        records.append({
            "lad_ons": code,
            "n_total": n_total,
            "n_miniroundabouts": n_minis,
            "n_raining": n_rain,
            "n_dark": n_dark,
            "n_dry": n_dry,
            "n_urban": n_urban,
            "n_slight": n_slight,
        })
    df = pd.DataFrame(records)
    df.to_csv(base / "la_collision.csv", index=False)
    print("Wrote", base / "la_collision.csv")


if __name__ == "__main__":
    make_energy_csv()
    make_la_energy_csv()
    make_la_collision_csv()
    print("All fake CSVs generated in data_cache/")
