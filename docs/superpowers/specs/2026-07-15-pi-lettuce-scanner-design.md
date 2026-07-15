# Raspberry Pi Lettuce Scanner — Design Spec

## Purpose

Deploy the trained lettuce-variant classifier to a Raspberry Pi with a physical shutter button, a Pi Camera Module, and a 16x2 I2C LCD. Pressing the button triggers a burst capture-and-classify cycle; the result is shown on the LCD.

## Scope

- New `raspberry_pi/` folder with the on-device script and its own requirements.
- A one-time conversion script (run on the dev machine, not the Pi) to regenerate `models/zenithV2.tflite` from the current `best_model.keras`.
- A correctness fix to `test7class.py` (missing BGR→RGB conversion — see Background).
- Out of scope: systemd auto-start (deferred — manual script run for now), any UI beyond the 16x2 LCD, retraining or model architecture changes.

## Background / prerequisite fix

While preparing this design, testing surfaced that `test7class.py` (and, by inheritance, any new Pi inference code) loads images via `cv2.imread`, which decodes in **BGR** channel order. Training (`train_7class.py`) used `image_dataset_from_directory`, which decodes in **RGB** order. Feeding BGR images to a model trained on RGB caused confidence swings of 20-40+ points on sample images and likely contributed to the Romaine/Others confusion noticed earlier during manual testing.

Fix: add `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` right after `cv2.imread` in `test7class.py`, and ensure the Pi script captures/feeds frames in RGB order from the start (see Camera component below — Picamera2's "RGB888" mode is a known gotcha where it can actually be BGR in memory depending on libcamera version, so this needs on-device verification once the hardware is available).

Separately confirmed: `mobilenet_v3.preprocess_input` is a no-op in the installed Keras version (scaling is already baked into the model's first `Lambda` layer), so no manual preprocessing call is needed anywhere in the Pi script.

## Architecture

```
MNetv3/
  raspberry_pi/
    lettuce_scanner.py   # main on-device script
    requirements.txt     # Pi-specific dependencies
    README.md            # wiring notes + run instructions + hardware smoke tests
  convert_to_tflite.py   # dev-machine only: best_model.keras -> models/zenithV2.tflite
```

`convert_to_tflite.py` runs on the development machine (where full TensorFlow is installed) since the Pi only needs the lightweight TFLite interpreter, not full TensorFlow.

## Dependencies (`raspberry_pi/requirements.txt`)

- `ai-edge-litert` — TFLite interpreter. **Not** `tflite-runtime`: the Pi runs Python 3.13, which `tflite-runtime` has no compatible wheel for (confirmed by prior hands-on attempt). Import as `from ai_edge_litert.interpreter import Interpreter`; the API (`allocate_tensors`, `get_input_details`, `set_tensor`, `invoke`, `get_tensor`) is a drop-in match for what `tflite_runtime` would have offered.
- `picamera2` — camera capture.
- `gpiozero` — shutter button input.
- `RPLCD` — 16x2 I2C LCD driver (HD44780-family over a PCF8574 backpack).
- `numpy`, `opencv-python-headless` (or `Pillow`) — resizing/array handling.

## Data flow

1. **Idle**: LCD shows `Press button to` / `scan lettuce`. Main loop blocks on `button.wait_for_press()`.
2. **On press**: LCD → `Processing...`.
3. Capture and classify 5 frames, ~300ms apart:
   - Capture a still frame via Picamera2.
   - Ensure RGB channel order (convert if the capture format turns out to be BGR — see Background).
   - Resize to 320×320.
   - Feed as float32, 0–255 range directly into the TFLite interpreter (no manual preprocessing).
   - Record `argmax` class index.
4. Majority vote across the 5 predictions. Confidence = (votes for the winning class) / 5.
5. Display:
   - If winner is `Others` → LCD shows `No Lettuce` / `Detected`.
   - Otherwise → class name on line 1, `Confidence: NN%` on line 2.
6. Hold the result on screen ~3 seconds, then return to idle.

## Components

### Model conversion (`convert_to_tflite.py`, dev machine)
- Standard float32 TFLite conversion from `best_model.keras` — no quantization. Keeps numeric behavior identical to the Keras model and avoids a separate calibration step; the interpreter's input/output tensors will be float32.

### Model loading
- `ai_edge_litert.interpreter.Interpreter(model_path="models/zenithV2.tflite")`.
- `allocate_tensors()` once at startup; cache input/output tensor details.
- Class label order: `["Butterhead", "Crisphead", "Looseleaf", "Oak", "Others", "RedLeaf", "Romaine"]` — matches `best_model.keras`'s current training-time alphabetical order (same list currently in `test7class.py`); TFLite conversion preserves output order.

### Camera
- `Picamera2`, configured for still capture.
- Explicit RGB verification/conversion step with a comment flagging the Picamera2 "RGB888-may-actually-be-BGR" gotcha for on-device confirmation (untestable from the dev environment).

### Button
- `gpiozero.Button(pin, bounce_time=0.1)`. Pin number as a top-of-file constant (default GPIO17).

### LCD
- `RPLCD.i2c.CharLCD`, I2C address and port expander as top-of-file constants (default address `0x27`).

## Error handling

- Camera capture failure: retry, up to 8 attempts total to gather 5 good frames; if still short, show `Camera Error` on the LCD and return to idle without crashing.
- Missing `models/zenithV2.tflite` at startup: clear console error, exit immediately (fail fast — this is a setup problem, not a runtime one).
- `Ctrl+C`: release camera and LCD resources cleanly before exiting.
- Transient I2C write errors: caught and logged to console; do not crash the scan loop.

## Testing

No physical Pi/camera/LCD hardware is available in the development environment, so:
- Model-loading, inference, and majority-vote logic will be written as plain functions, verifiable against a static test image (e.g. `test_img/looseleaf1.jpg`) independent of any hardware.
- `raspberry_pi/README.md` will include standalone smoke tests for camera, button, and LCD individually, to be run on the Pi before the first full integrated run.

## Explicitly deferred

- systemd service / auto-start on boot.
- Any display beyond the 16x2 LCD.
- Anything reordering or retraining the model itself.
