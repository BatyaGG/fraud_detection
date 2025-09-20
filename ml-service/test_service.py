# test_client.py
import os
import json
import requests
import math
import pandas as pd
from datetime import date

# URL = os.getenv("URL", "http://207.180.201.244:8000")
URL = os.getenv("URL", "http://localhost:8000")
K = int(os.getenv("TOPK", "8"))


def csv_to_payload(csv_path: str) -> dict:
    def nan2none(v):
        return None if (pd.isna(v) or (isinstance(v, float) and math.isnan(v))) else v

    def parse_dt(val, out_fmt):
        if pd.isna(val):
            return None
        try:
            return pd.to_datetime(val).strftime(out_fmt)
        except Exception:
            return None

    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        po = nan2none(r.get("Purchase Order"))
        fy = nan2none(r.get("Fiscal Year"))

        row = {
            "Fiscal Year": int(fy) if fy is not None else None,
            "Purchase Order Date": parse_dt(r.get("Purchase Order Date"), "%Y-%m-%d"),
            "Purchase Order": int(po) if po is not None else None,
            "Purchase Order Line": None,
            "Contract Number": nan2none(r.get("Contract Number")),
            "Contract Title": nan2none(r.get("Contract Title")),
            "Purchasing Department": nan2none(r.get("Purchasing Department")),
            "Purchasing Department Title": nan2none(r.get("Purchasing Department Title")),
            "Post Date - Original": None,
            "Post Date - Current": None,
            "Commodity Code": nan2none(r.get("Commodity Code")),
            "Commodity Title": nan2none(r.get("Commodity Title")),
            "Supplier & Other Non-Supplier Payees": nan2none(r.get("Supplier & Other Non-Supplier Payees")),
            "Supplier Street": None,
            "Supplier City": nan2none(r.get("Supplier City")),
            "Supplier State": nan2none(r.get("Supplier State")),
            "Supplier ZIP Code": nan2none(r.get("Supplier ZIP Code")),
            "Supplier Contact": None,
            "Supplier Email": nan2none(r.get("Supplier Email")),
            "Supplier Phone": nan2none(r.get("Supplier Phone")),
            "Encumbered Quantity": nan2none(r.get("Encumbered Quantity")),
            "Encumbered Amount": nan2none(r.get("Encumbered Amount")),
            "data_as_of": None,
            "data_loaded_at": None,
            "PO6": int(po) if po is not None else None,
            "PO6_key": int(po) if po is not None else None,
            "vouchers_paid_sum": nan2none(r.get("vouchers_paid_sum")),
            "vouchers_pending_sum": nan2none(r.get("vouchers_pending_sum")),
            "encum_balance_sum": nan2none(r.get("encum_balance_sum")),
            "vouchers_count": nan2none(r.get("vouchers_count")),
            "first_voucher_date": parse_dt(r.get("first_voucher_date"), "%Y-%m-%d %H:%M:%S"),
            "last_voucher_date": parse_dt(r.get("last_voucher_date"), "%Y-%m-%d %H:%M:%S"),
        }
        rows.append(row)

    return {"rows": rows}


def iso(d: date):
    return d.isoformat()


def post_json(path, payload):
    r = requests.post(f"{URL}{path}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def get_json(path):
    r = requests.get(f"{URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    print(f"[i] Health: {get_json('/health')}")
    print(f"[i] Version: {get_json('/version')}")

    # payload = json.load(open("test_payload.json"))
    # rows = payload["rows"]
    import time
    s = time.time()
    for i in range(100):
        payload = csv_to_payload("po_vendor_x.csv")
        res = post_json("/score", payload)

        with open("res.json", "w") as f:
            json.dump(res, f, indent=2)

    avg = (time.time() - s) / 100
    print(f"[i] Average time: {avg:.2f}s")
    print()


if __name__ == "__main__":
    main()
