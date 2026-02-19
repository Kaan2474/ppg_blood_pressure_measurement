import sys
import asyncio
import csv
import time
import struct
from bleak import BleakScanner, BleakClient
from bleakheart import PolarMeasurementData

INPUT_FILE = "signals_walking_2.csv"


# Writes the headers into the CSV file
with open(INPUT_FILE, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Computer Timestamp", "Polar Timestamp", "Type", 
                     "Val0", "Val1", "Val2", "Val3"])


# --- HELPER FUNCTIONS ---

def save_to_csv(polar_timestamp, data_type, samples):
    """
    Stores PPG and ACC data into a single CSV file

    Args:
        polar_timestamp: Internal time of the Polar Verity Sense device
        data_type: Either PPG or ACC
        samples: Either PPG or ACC signals
    """
    computer_timestamp = time.time_ns()

    with open(INPUT_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        for sample in samples:
            row = [computer_timestamp, polar_timestamp, data_type] + list(sample)
            # PPG = 4 values, ACC = 3 values -> Missing value for ACC filled with empty string
            while len(row) < 7: 
                row.append("")
            writer.writerow(row)


def decode_polar_acc(raw_acc_signals):
    """
    Decodes the ACC signals from bytes into real numbers

    Args:
        raw_acc_signals: ACC signals in bytearray format
    
    Returns:
        decoded_signals: Decoded ACC signals 
    """
    if len(raw_acc_signals) < 20: return [] # Valid ACC signals contain at least 20 bytes.
    
    '''
    - Raw data for the actual accelerometer readings starts at Byte 10.
    - unpack_from reads binary data
    - Starting X, Y, and Z values (e.g., 1000, -50, 980)
    '''
    # Reads raw, binary ACC signals, which start at byte 10
    x_ref, y_ref, z_ref = struct.unpack_from('<hhh', raw_acc_signals, 10)
    decoded_signals = [(x_ref, y_ref, z_ref)]
    
    delta_bits = raw_acc_signals[16] 
    sample_count = raw_acc_signals[17]
    
    bit_offset = 0
    byte_start = 18
    current_x, current_y, current_z = x_ref, y_ref, z_ref
    
    for _ in range(sample_count):
        deltas = []
        for _ in range(3): # X, Y, Z
            val = 0
            bits_read = 0
            while bits_read < delta_bits:
                byte_idx = byte_start + ((bit_offset) // 8)
                if byte_idx >= len(raw_acc_signals): break
                bit_in_byte = (bit_offset) % 8
                bits_to_take = min(delta_bits - bits_read, 8 - bit_in_byte)
                mask = (1 << bits_to_take) - 1
                chunk = (raw_acc_signals[byte_idx] >> bit_in_byte) & mask
                val |= (chunk << bits_read)
                bits_read += bits_to_take
                bit_offset += bits_to_take
            
            # Sign extension
            if val >= (1 << (delta_bits - 1)):
                val -= (1 << delta_bits)
            deltas.append(val)
        
        if len(deltas) == 3:
            current_x += deltas[0]
            current_y += deltas[1]
            current_z += deltas[2]
            decoded_signals.append((current_x, current_y, current_z))
            
    return decoded_signals


# --- CALLBACKS ---

def accel_callback(data):
    # data: ('ACC', timestamp, bytearray)
    if len(data) < 3: return
    data_type = data[0]
    timestamp = data[1]
    raw_bytes = data[2] # e.g., \x02\x84\x94
    decoded_acc_signals = decode_polar_acc(raw_bytes) # Decodes the bytearrays into real numbers
    if decoded_acc_signals:
        save_to_csv(timestamp, data_type, decoded_acc_signals)


def ppg_callback(data):
    # data: ('PPG', timestamp, [[ch0, ch1, ch2, amb], ...])
    if len(data) < 3: return
    data_type = data[0]
    timestamp = data[1]
    ppg_signals = data[2]
    if ppg_signals:
        save_to_csv(timestamp, data_type, ppg_signals)


# --- MAIN ASYNC LOOP ---

# Handle OS differences regarding the keyboard
if sys.platform=="win32":
    from threading import Thread
    add_reader_support=False
else:
    add_reader_support=True


async def scan():
    """
    Searches for the Polar Verity Sense device

    Returns:
        device: The found Polar Verity Sense device
    """
    device = await BleakScanner.find_device_by_filter(
        lambda dev, adv: dev.name and "polar" in dev.name.lower())
    return device


async def run_ble_client(device):
    """
    The function has multiple tasks:
        - Keeps running until 'Enter' is pressed or device disconnects
        - Establishes a connection between the Polar Verity Sense device and the computer
        - Handles incoming PPG and ACC data
        - Starts and stops the data streaming of PPG and ACC
    
    Args:
        device: The Polar Verity Sense device
    """

    quitclient = asyncio.Event() # Flag for quitting the program, initially set to False

    # Stops the program when 'Enter' is pressed
    def keyboard_handler(loop=None):
        input() 
        print("Quitting...")
        if loop is None:
            quitclient.set() # Sets the quitclient flag to True
        else:
            loop.call_soon_threadsafe(quitclient.set)    

    # Stops the program when the bluetooth connection drops
    def disconnected_callback():
        print("Sensor disconnected")
        quitclient.set() # Sets the quitclient flag to True

    # Connects to the Polar Verity Sense using a bluetooth connection
    async with BleakClient(device, disconnected_callback=disconnected_callback) as client:
        print(f"Connected: {client.is_connected}")
        
        # Handles arrivin data streams
        def master_callback(data):
            # If incoming signals are ACC execute accel_callback
            if data[0] == 'ACC':
                accel_callback(data)
            # If incoming signals are PPG execute ppg_callback
            elif data[0] == 'PPG':
                ppg_callback(data)
            else:
                print(f"Unknown data type: {data[0]}")

        pmd = PolarMeasurementData(client, callback=master_callback) # Object for data streaming and stoping
        
        await pmd.start_streaming('SDK')
        print("Starting Streams...")
        await pmd.start_streaming('ACC', RANGE=8, SAMPLE_RATE=52, CHANNELS=3)
        await pmd.start_streaming('PPG', SAMPLE_RATE=176, RESOLUTION=22, CHANNELS=4) 
        
        # Handle Input
        loop = asyncio.get_running_loop()
        # Input when OS = MACOS
        if add_reader_support:
            loop.add_reader(sys.stdin, keyboard_handler)
        # Input when OS = Windows
        else:
            Thread(target=keyboard_handler, kwargs={'loop': loop}, daemon=True).start()
                    
        await quitclient.wait() # Program stops here until quitclient flag is set to True
        
        # Before quitting the program, the data stream is stopped
        if client.is_connected:
            await pmd.stop_streaming('ACC')
            await pmd.stop_streaming('PPG')
            
        if add_reader_support:
            loop.remove_reader(sys.stdin)


async def main():
    print("Scanning...")
    device = await scan()
    if not device:
        print("No device found.")
        sys.exit(-1)
        
    await run_ble_client(device)
    print("Bye.")


if __name__ == "__main__":
    asyncio.run(main())