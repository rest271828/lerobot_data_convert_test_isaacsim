# ws_sink.py
import asyncio, websockets

async def handler(ws):
    print("client connected:", ws.remote_address)
    while True:
        msg = await ws.recv()
        if isinstance(msg, (bytes, bytearray)):
            print("got binary bytes:", len(msg))
        else:
            print("got text:", msg[:200])

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000, max_size=2**28):
        print("ws server listening on ws://0.0.0.0:8000")
        await asyncio.Future()

asyncio.run(main())
