# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import os
import threading
import uuid
from collections import defaultdict

import aiohttp

from quart import Quart, make_response, request, jsonify

count = 0
prefill_decode_urls = []
prefill_decode_cv = threading.Condition()
encoder_urls = []

_encoder_inflight: dict[str, int] = defaultdict(int)
_encoder_lock = asyncio.Lock()

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content

MM_TYPES = {"image_url", "audio_url", "input_audio"}

def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items

async def _pick_encoder(e_urls: list[str]) -> str:
    """Return the URL of the encoder instance with the fewest in-flight requests."""
    async with _encoder_lock:
        url = min(e_urls, key=lambda u: _encoder_inflight[u])
        _encoder_inflight[url] += 1
        return url

async def _release_encoder(url: str) -> None:
    async with _encoder_lock:
        _encoder_inflight[url] = max(0, _encoder_inflight[url] - 1)

async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
) -> None:
    """
    1. Build one request *per MM item* with all text removed.
    2. Pick the least-loaded encoder for each item (load balancing).
    3. Send them all concurrently to the encode cluster (fanout).
    4. Raise if any of them fails.
    """
    print("[%s] Processing multimodal items...", req_id)
    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        print("[%s] No multimodal items, skipping encoder", req_id)
        return
    print("[%s] got %d multimodal items...", req_id, len(mm_items))

    async def post_one(session: aiohttp.ClientSession, item: dict, idx: int) -> bytes:
        child_req_id = f"{req_id}:{idx}"
        target_url = await _pick_encoder(e_urls)
        try:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "X-Request-Id": child_req_id,
            }
            data = {
                "model": orig_request.get("model"),
                "messages": [{"role": "user", "content": [item]}],
                "max_tokens": 1,
                "stream": False,
            }
            url = f"{target_url}/v1/chat/completions"
            async with session.post(url=url, json=data, headers=headers) as response:
                body = await response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"Encoder primer failed for item {idx}: "
                        f"HTTP {response.status} — {body.decode(errors='replace')}"
                    )
                return body
        finally:
            await _release_encoder(target_url)  # always decrement, even on failure

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        tasks = [post_one(session, item, idx) for idx, item in enumerate(mm_items)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(f"Encoder primer had {len(errors)} failure(s): {errors}")
    print("[%s] All %d encoder requests completed successfully", req_id, len(mm_items))


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_decode_urls
        global prefill_decode_cv
        with prefill_decode_cv:
            prefill_decode_addr = prefill_decode_urls[count % len(prefill_decode_urls)]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_decode_addr}"
        )
        count += 1

        request_id = random_uuid()

        global encoder_urls
        await fanout_encoder_primer(original_request_data, encoder_urls, request_id)


        generator = forward_request(
            f"{prefill_decode_addr}{request.path}", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
async def health_check():
    """
    Checks the health of the proxy and the underlying instances.
    """
    global prefill_decode_urls
    
    # 1. Check if we have any instances registered at all
    if not prefill_decode_urls:
        return {"status": "unhealthy", "reason": "No instances registered"}, 503

    # 2. Try to forward the health check to the first available prefill instance
    try:
        # Get the first available address
        target_addr = prefill_decode_urls[0]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{target_addr}/health") as response:
                if response.status == 200:
                    return {"status": "healthy", "remote_status": 200}, 200
                else:
                    return {"status": "degraded", "remote_status": response.status}, 502
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-decode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )

    args = parser.parse_args()
    encoder_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    prefill_decode_urls = [
        u.strip() for u in args.prefill_decode_servers_urls.split(",") if u.strip()
    ]

    app.run(host=args.host, port=10001)
