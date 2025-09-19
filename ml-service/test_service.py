# test_client.py
import os
import json
import requests
from datetime import date, timedelta

# URL = os.getenv("URL", "http://207.180.201.244:8000")
URL = os.getenv("URL", "http://localhost:8000")
K = int(os.getenv("TOPK", "8"))


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

    payload = json.load(open("test_payload.json"))
    rows = payload["rows"]
    res = post_json("/score", payload)

    print(res)



if __name__ == "__main__":
    main()
