# Lettuce Scanner (Raspberry Pi)

Press a button, capture and classify 5 frames of a lettuce leaf, show the result on a 16x2 I2C LCD.

## Hardware

- Raspberry Pi with a Pi Camera Module (CSI ribbon)
- Push button wired to GPIO17 and ground
- 16x2 character LCD on a PCF8574 I2C backpack (default address `0x27`)

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure `../models/zenithV2.tflite` exists (generate it from the repo root with `python convert_to_tflite.py` if it doesn't).

Enable the camera and I2C interfaces if you haven't already:

```
sudo raspi-config
# Interface Options -> Camera -> Enable
# Interface Options -> I2C -> Enable
```

Confirm the LCD is detected at the expected address:

```
i2cdetect -y 1
```

If it shows up at an address other than `0x27`, update `LCD_I2C_ADDRESS` in `lettuce_scanner.py`.

## Smoke tests before the first full run

Run these individually so a wiring issue in one component doesn't get confused for a bug in the whole script.

**Camera:**
```python
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"format": "RGB888"}))
picam2.start()
frame = picam2.capture_array()
print(frame.shape)  # should be (height, width, 3)
```
Save a frame to a file and open it to confirm colors look right (green lettuce should look green, not blue/orange). If colors are inverted, the BGR→RGB conversion in `capture_rgb_frame()` needs to be removed instead of applied — Picamera2's "RGB888" format is inconsistently BGR or RGB depending on libcamera version.

**Button:**
```python
from gpiozero import Button
button = Button(17, bounce_time=0.1)
print("Press the button...")
button.wait_for_press()
print("Pressed!")
```

**LCD:**
```python
from RPLCD.i2c import CharLCD
lcd = CharLCD("PCF8574", 0x27, cols=16, rows=2)
lcd.write_string("Hello, lettuce!")
```

## Run

```
python lettuce_scanner.py
```

Ctrl+C to stop.
