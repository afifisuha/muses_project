#Commands Bytes
# Torque (not yet supported), time, activated, direction.
CLOSE_COMMAND = [0,20,0b01111000,0b01111000]
OPEN_COMMAND = [0,20,0b01111000,0b00000000]
LEFT_COMMAND = [0,20,0b10000000,0b10000000]
RIGHT_COMMAND = [0,20,0b10000000,0b00000000]

commands = {
    "open": [5] + OPEN_COMMAND,
    "close": [5] + CLOSE_COMMAND,
    "left": [5] + LEFT_COMMAND,
    "right": [5] + RIGHT_COMMAND
}
DIRECT_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"
HAND_MAC = "B8:D6:1A:40:EF:D6"