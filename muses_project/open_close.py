import asyncio
from protocol import HAND_MAC, DIRECT_UUID, commands
from bleak import BleakClient
from time import sleep

async def move(movement: str):
    async with BleakClient(HAND_MAC) as client:
        await client.write_gatt_char(DIRECT_UUID, bytes(commands[movement]))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(move("open"))
    loop.run_until_complete(move("close"))
