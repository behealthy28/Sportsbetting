#!/usr/bin/env python3
"""Run this to diagnose Polymarket API responses."""
import requests
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://polymarket.com/",
    "Origin": "https://polymarket.com",
}

print("=" * 60)
print("TEST 1: gamma API /markets (sports tag)")
print("=" * 60)
try:
    r = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "closed": "false", "limit": 5,
                "tag_slug": "sports", "order": "volumeNum", "ascending": "false"},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        print(f"Markets returned: {len(markets)}")
        for m in markets[:3]:
            print(f"  Q: {m.get('question','')[:70]}")
            print(f"  endDate: {m.get('endDate') or m.get('end_date_iso','')}")
            print(f"  category: {m.get('category','')}")
            print(f"  tags: {m.get('tags','')}")
            print()
    else:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 2: gamma API /markets (no tag filter)")
print("=" * 60)
try:
    r = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "closed": "false", "limit": 5,
                "order": "volumeNum", "ascending": "false"},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        print(f"Markets returned: {len(markets)}")
        for m in markets[:3]:
            print(f"  Q: {m.get('question','')[:70]}")
            print(f"  category: {m.get('category','')}")
            print()
    else:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 3: CLOB API /markets")
print("=" * 60)
try:
    r = requests.get(
        "https://clob.polymarket.com/markets",
        params={"active": "true", "limit": 5},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        markets = data.get("data", []) if isinstance(data, dict) else data
        print(f"Markets returned: {len(markets)}")
        for m in markets[:3]:
            print(f"  Q: {m.get('question','')[:70]}")
            print(f"  end_date_iso: {m.get('end_date_iso','')}")
            print(f"  category: {m.get('category','')}")
            print()
    else:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST 4: gamma API - raw first result keys")
print("=" * 60)
try:
    r = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "limit": 1},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        if markets:
            print("Keys in first market:")
            print(json.dumps(list(markets[0].keys()), indent=2))
            print("Full first market:")
            print(json.dumps(markets[0], indent=2, default=str)[:1000])
except Exception as e:
    print(f"ERROR: {e}")
