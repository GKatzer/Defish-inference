"""How long GET /health takes while 8 requests of /detect (or /analyze) are being processed at the same time.

usage: python docs/examples/health_under_load.py http://127.0.0.1:8000 docs/examples/photos/planted-tank-tetras.jpg detect|analyze
Start the service with RATE_LIMIT_REQUESTS raised (e.g. 1000). Needs httpx (requirements.txt).
"""
import asyncio
import base64
import sys
import time

import httpx

BASE, PHOTO, ENDPOINT = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(PHOTO, "rb").read()


async def heavy(client):
    if ENDPOINT == "detect":
        await client.post(BASE + "/detect", files={"image": ("p.jpg", raw, "image/jpeg")})
    else:
        await client.post(BASE + "/analyze", json={"image_bytes": base64.b64encode(raw).decode()})


async def poll_health(client, latencies, stop):
    while not stop.is_set():
        started = time.perf_counter()
        await client.get(BASE + "/health")
        latencies.append((time.perf_counter() - started) * 1000)
        await asyncio.sleep(0.01)


async def main():
    async with httpx.AsyncClient(timeout=120) as client:
        await client.get(BASE + "/health")
        idle, stop = [], asyncio.Event()
        task = asyncio.create_task(poll_health(client, idle, stop))
        await asyncio.sleep(1)
        stop.set()
        await task
        busy, stop = [], asyncio.Event()
        task = asyncio.create_task(poll_health(client, busy, stop))
        await asyncio.gather(*[heavy(client) for _ in range(8)])
        stop.set()
        await task
        median = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"/health while idle:                    median {median(idle):7.1f} ms, max {max(idle):7.1f} ms ({len(idle)} calls)")
        print(f"/health during 8 concurrent /{ENDPOINT}: median {median(busy):7.1f} ms, max {max(busy):7.1f} ms ({len(busy)} calls)")


asyncio.run(main())
