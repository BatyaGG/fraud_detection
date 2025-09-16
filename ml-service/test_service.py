# test_client.py
import os
import json
import requests
from datetime import date, timedelta

URL = os.getenv("URL", "http://207.180.201.244:8000")
K = int(os.getenv("TOPK", "8"))


def iso(d: date):
    return d.isoformat()


def build_rows():
    """
    Believable procurement-ish sample:
    - Normal POs/invoices
    - Duplicate invoice (same vendor+invoice_number+amount, later date)
    - Split pattern (3 invoices just under a 10k threshold, same day)
    - High-amount (likely 'price gouging' vibe)
    - Mixed dates to exercise step/time deltas
    """
    base = date(2025, 3, 10)

    rows = [
        # Normal small supply
        dict(vendor="ACME SUPPLIES LLC", po_number="PO-2025-0001", invoice_number="INV-1001",
             qty=100, unit_price=1.20, amount=120.0,
             po_date=iso(base - timedelta(days=7)),
             invoice_date=iso(base - timedelta(days=5)),
             payment_date=iso(base - timedelta(days=2)), type="PAYMENT"),

        # Normal IT equipment
        dict(vendor="BRIGHT IT CO", po_number="PO-2025-0002", invoice_number="INV-2001",
             qty=15, unit_price=850.0, amount=12750.0,
             po_date=iso(base - timedelta(days=12)),
             invoice_date=iso(base - timedelta(days=9)),
             payment_date=None, type="TRANSFER"),

        # DUPLICATE invoice #INV-874 (first occurrence)
        dict(vendor="OFFICEMAXX LTD", po_number="PO-2025-0010", invoice_number="INV-874",
             qty=1, unit_price=124500.0, amount=124500.0,
             po_date=iso(base - timedelta(days=20)),
             invoice_date=iso(base - timedelta(days=18)),
             payment_date=iso(base - timedelta(days=15)), type="PAYMENT"),

        # DUPLICATE invoice #INV-874 (second occurrence, later date)
        dict(vendor="OFFICEMAXX LTD", po_number="PO-2025-0010", invoice_number="INV-874",
             qty=1, unit_price=124500.0, amount=124500.0,
             po_date=iso(base - timedelta(days=20)),
             invoice_date=iso(base - timedelta(days=9)),  # later invoice date
             payment_date=None, type="PAYMENT"),

        # SPLIT pattern around approval threshold = 10,000
        dict(vendor="NORTHSHORE MEDICAL", po_number="PO-2025-0020", invoice_number="INV-3001",
             qty=1, unit_price=9950.0, amount=9950.0,
             po_date=iso(base - timedelta(days=2)),
             invoice_date=iso(base - timedelta(days=1)),
             payment_date=None, type="PAYMENT"),

        dict(vendor="NORTHSHORE MEDICAL", po_number="PO-2025-0020", invoice_number="INV-3002",
             qty=1, unit_price=9980.0, amount=9980.0,
             po_date=iso(base - timedelta(days=2)),
             invoice_date=iso(base - timedelta(days=1)),
             payment_date=None, type="PAYMENT"),

        dict(vendor="NORTHSHORE MEDICAL", po_number="PO-2025-0020", invoice_number="INV-3003",
             qty=1, unit_price=9925.0, amount=9925.0,
             po_date=iso(base - timedelta(days=2)),
             invoice_date=iso(base - timedelta(days=1)),
             payment_date=None, type="PAYMENT"),

        # High-amount network gear (likely to trigger high log_amount hint)
        dict(vendor="BRIGHT IT CO", po_number="PO-2025-0030", invoice_number="INV-5007",
             qty=200, unit_price=320.0, amount=64000.0,
             po_date=iso(base - timedelta(days=14)),
             invoice_date=iso(base - timedelta(days=10)),
             payment_date=None, type="PAYMENT"),

        # Construction services (very high amount)
        dict(vendor="URBAN BUILD INC", po_number="PO-2025-0042", invoice_number="INV-9200",
             qty=1, unit_price=250000.0, amount=250000.0,
             po_date=iso(base - timedelta(days=30)),
             invoice_date=iso(base - timedelta(days=23)),
             payment_date=None, type="TRANSFER"),

        # A couple more normal lines to mix distribution
        dict(vendor="ACME SUPPLIES LLC", po_number="PO-2025-0003", invoice_number="INV-1010",
             qty=250, unit_price=0.95, amount=237.50,
             po_date=iso(base - timedelta(days=4)),
             invoice_date=iso(base - timedelta(days=2)),
             payment_date=None, type="PAYMENT"),

        dict(vendor="CITY CLEANERS", po_number="PO-2025-0060", invoice_number="INV-7001",
             qty=12, unit_price=75.0, amount=900.0,
             po_date=iso(base - timedelta(days=8)),
             invoice_date=iso(base - timedelta(days=6)),
             payment_date=iso(base - timedelta(days=3)), type="PAYMENT"),
    ]

    # Add a few small random-ish “office supply” rows to make list longer
    for i in range(8):
        amt = 50 + 25 * i
        rows.append(dict(
            vendor="ACME SUPPLIES LLC", po_number=f"PO-2025-01{i:02d}",
            invoice_number=f"INV-1{i:03d}",
            qty=10 + i, unit_price=amt / (10 + i), amount=float(amt),
            po_date=iso(base - timedelta(days=10 + i)),
            invoice_date=iso(base - timedelta(days=8 + i)),
            payment_date=None, type="PAYMENT"
        ))

    return rows


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

    rows = build_rows()
    payload = {"rows": rows}
    res = post_json("/score", payload)

    print("\n=== Raw response (first item) ===")
    print(json.dumps({k: (v if k != "scores" else v[:1]) for k, v in res.items()}, indent=2))

    # Build a handy table with vendor/invoice/amount and model results
    by_id = {i: r for i, r in enumerate(rows)}
    scored = []
    for s in res["scores"]:
        row = by_id.get(s["row_id"], {})
        scored.append({
            "row_id": s["row_id"],
            "vendor": row.get("vendor"),
            "invoice": row.get("invoice_number"),
            "amount": row.get("amount"),
            "ml_score": s["ml_score"],
            "anomaly_score": s["anomaly_score"],
            "reason_hint": s["reason_hint"],
            "top_features": s["top_features"],
        })

    # Sort by ml_score desc and print Top-K
    scored.sort(key=lambda x: x["ml_score"], reverse=True)
    print(f"\n=== Top-{K} by ml_score ===")
    for i, it in enumerate(scored[:K], 1):
        tf = ", ".join([f"{n}({c:+.2f})" for n, c in it["top_features"]])
        print(f"{i:02d}. row={it['row_id']:>2}  vendor={it['vendor']:<20} "
              f"inv={it['invoice']:<10} amount={it['amount']:>10,.2f}  "
              f"p={it['ml_score']:.3f}  hint=[{it['reason_hint']}]  top=[{tf}]")


if __name__ == "__main__":
    main()
