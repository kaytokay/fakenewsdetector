"""Bounded live telemetry stream endpoint."""
import asyncio, json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..state import live_analytics, telemetry_connections
router = APIRouter()
@router.get("/stream-stats")
async def stream_stats():
    """Serve aggregate session telemetry with a bounded number of SSE clients."""
    if not telemetry_connections.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Telemetry stream is at capacity. Try again shortly.")

    async def generator():
        last = -1
        last_sent = asyncio.get_running_loop().time()
        try:
            while True:
                if live_analytics["total_analyzed"] != last:
                    last = live_analytics["total_analyzed"]
                    yield f"data: {json.dumps(live_analytics)}\n\n"
                    last_sent = asyncio.get_running_loop().time()
                elif asyncio.get_running_loop().time() - last_sent >= 15:
                    yield ": keep-alive\n\n"
                    last_sent = asyncio.get_running_loop().time()
                await asyncio.sleep(1)
        finally:
            telemetry_connections.release()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

