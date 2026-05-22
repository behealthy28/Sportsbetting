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
print("TEST A: gamma API — limit=100, sorted by endDateIso asc")
print("(active+closed=false should skip expired markets)")
print("=" * 60)
try:
    r = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "closed": "false", "limit": 100,
                "order": "endDateIso", "ascending": "true"},
        headers=HEADERS, timeout=15,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        print(f"Markets returned: {len(markets)}")
        print("First 5:")
        for m in markets[:5]:
            print(f"  Q: {m.get('question','')[:70]}")
            print(f"  endDateIso: {m.get('endDateIso','')}  endDate: {m.get('endDate','')[:25]}")
            print()
        print("Last 3:")
        for m in markets[-3:]:
            print(f"  Q: {m.get('question','')[:70]}")
            print(f"  endDateIso: {m.get('endDateIso','')}")
    else:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST B: gamma API — offset pagination (offset=0, 500, 1000)")
print("=" * 60)
for offset in [0, 500, 1000]:
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 5,
                    "offset": offset, "order": "endDateIso", "ascending": "true"},
            headers=HEADERS, timeout=12,
        )
        print(f"offset={offset}  status={r.status_code}", end="  ")
        if r.status_code == 200:
            data = r.json()
            markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
            print(f"returned={len(markets)}")
            for m in markets[:2]:
                print(f"    Q: {m.get('question','')[:65]}")
                print(f"    endDateIso: {m.get('endDateIso','')}")
        else:
            print(r.text[:100])
    except Exception as e:
        print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST C: CLOB API — show next_cursor in response")
print("=" * 60)
try:
    r = requests.get(
        "https://clob.polymarket.com/markets",
        params={"limit": 10},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        if isinstance(data, dict):
            print(f"next_cursor: {data.get('next_cursor', 'NOT FOUND')}")
            print(f"count: {data.get('count', 'N/A')}")
            markets = data.get("data", [])
            print(f"Markets in data: {len(markets)}")
            if markets:
                print(f"First market end_date_iso: {markets[0].get('end_date_iso','')}")
                print(f"Last  market end_date_iso: {markets[-1].get('end_date_iso','')}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST D: CLOB page 2 (using cursor from TEST C)")
print("=" * 60)
try:
    r1 = requests.get("https://clob.polymarket.com/markets",
                      params={"limit": 500}, headers=HEADERS, timeout=12)
    if r1.status_code == 200:
        d1 = r1.json()
        cursor = d1.get("next_cursor", "") if isinstance(d1, dict) else ""
        print(f"Cursor from page 1: {cursor!r}")
        if cursor:
            r2 = requests.get("https://clob.polymarket.com/markets",
                              params={"limit": 500, "next_cursor": cursor},
                              headers=HEADERS, timeout=12)
            if r2.status_code == 200:
                d2 = r2.json()
                markets2 = d2.get("data", []) if isinstance(d2, dict) else d2
                print(f"Page 2 markets: {len(markets2)}")
                print(f"Page 2 cursor: {d2.get('next_cursor','') if isinstance(d2, dict) else 'N/A'!r}")
                if markets2:
                    print(f"First end_date: {markets2[0].get('end_date_iso','')}")
                    print(f"Last  end_date: {markets2[-1].get('end_date_iso','')}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("TEST E: gamma events API")
print("=" * 60)
try:
    r = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"active": "true", "closed": "false", "limit": 10,
                "order": "volumeNum", "ascending": "false"},
        headers=HEADERS, timeout=12,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        events = data if isinstance(data, list) else data.get("data", data.get("events", []))
        print(f"Events returned: {len(events)}")
        for ev in events[:5]:
            print(f"  title: {ev.get('title','')[:60]}")
            print(f"  slug:  {ev.get('slug','')[:60]}")
            mkts = ev.get("markets", [])
            print(f"  markets inside: {len(mkts)}")
            if mkts:
                print(f"    first Q: {mkts[0].get('question','')[:60]}")
            print()
    else:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")
