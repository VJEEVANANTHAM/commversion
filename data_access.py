"""Deterministic property and seller-coverage tools."""
from pathlib import Path
import re
import pandas as pd

def normalize_postcode(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()

class PropertyData:
    def __init__(self, data_dir="data"):
        d=Path(data_dir)
        self.properties=pd.read_csv(d/"properties_for_buyers.csv").fillna("")
        self.sellers={normalize_postcode(x) for x in pd.read_csv(d/"seller_coverage.csv")["postcode"].tolist()}
        self.properties["postcode_norm"]=self.properties["postcode"].map(normalize_postcode)
        self.properties["category_norm"]=self.properties["listing_category"].astype(str).str.lower().str.strip()
        self.properties["status_norm"]=self.properties["status"].astype(str).str.lower().str.strip()

    def at_postcode(self, postcode, category):
        return self.properties[(self.properties.postcode_norm==normalize_postcode(postcode)) &
            (self.properties.category_norm==category)].copy()

    def available_matches(self, postcode, category, budget=None):
        rows=self.at_postcode(postcode, category)
        rows=rows[rows.status_norm=="available"]
        if category=="new_build" and budget is not None:
            rows=rows[pd.to_numeric(rows.price_gbp, errors="coerce")<=budget]
        return rows.sort_values("price_gbp").head(5)

    def is_seller_covered(self, postcode):
        return normalize_postcode(postcode) in self.sellers

    def find_property(self, query):
        q=str(query).strip().lower()
        if not q: return self.properties.iloc[0:0].copy()
        p=self.properties
        mask=(p["property_name"].astype(str).str.lower().str.contains(re.escape(q),na=False) |
              p["address"].astype(str).str.lower().str.contains(re.escape(q),na=False) |
              p["property_id"].astype(str).str.lower().str.contains(re.escape(q),na=False))
        return p[mask].copy()

    @staticmethod
    def money(value):
        try: return "£{:,.0f}".format(float(value))
        except (ValueError,TypeError): return str(value)
