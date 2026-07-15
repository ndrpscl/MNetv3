import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from gpiozero import Button
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "zenithV2.tflite"
CLASS_NAMES = ["Butterhead", "Crisphead", "Looseleaf", "Oak", "Others", "RedLeaf", "Romaine"]
IMAGE_SIZE = (320, 320)

BUTTON_PIN = 17
LCD_I2C_EXPANDER = "PCF8574"
LCD_I2C_ADDRESS = 0x27

NUM_POLLS = 5
POLL_INTERVAL_S = 0.3
MAX_CAPTURE_ATTEMPTS = 8
RESULT_HOLD_S = 3


def load_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def capture_rgb_frame(picam2):
    frame = picam2.capture_array()
    # Picamera2's "RGB888" stream format is actually BGR-ordered in memory
    # on some libcamera versions. Verify against a known-color subject on
    # real hardware; drop this conversion if colors already come out right.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def classify_frame(interpreter, frame_rgb):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    resized = cv2.resize(frame_rgb, IMAGE_SIZE)
    input_data = np.expand_dims(resized, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return int(np.argmax(output))


def poll_and_vote(interpreter, picam2):
    votes = []
    attempts = 0
    while len(votes) < NUM_POLLS and attempts < MAX_CAPTURE_ATTEMPTS:
        attempts += 1
        try:
            frame = capture_rgb_frame(picam2)
            votes.append(classify_frame(interpreter, frame))
        except Exception as exc:
            print(f"Capture/classify failed (attempt {attempts}): {exc}")
            continue
        time.sleep(POLL_INTERVAL_S)

    if len(votes) < NUM_POLLS:
        return None, 0.0

    winner, count = Counter(votes).most_common(1)[0]
    return winner, count / len(votes)


def show_idle(lcd):
    lcd.clear()
    lcd.write_string("Press button to")
    lcd.cursor_pos = (1, 0)
    lcd.write_string("scan lettuce")


def show_processing(lcd):
    lcd.clear()
    lcd.write_string("Processing...")


def show_result(lcd, class_index, confidence):
    lcd.clear()
    if class_index is None:
        lcd.write_string("Camera Error")
        return

    label = CLASS_NAMES[class_index]
    if label == "Others":
        lcd.write_string("No Lettuce")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Detected")
    else:
        lcd.write_string(label)
        lcd.cursor_pos = (1, 0)
        lcd.write_string(f"Confidence: {confidence * 100:.0f}%")


def main():
    try:
        interpreter = load_interpreter(MODEL_PATH)
    except Exception as exc:
        print(f"Failed to load model at {MODEL_PATH}: {exc}")
        return

    lcd = CharLCD(LCD_I2C_EXPANDER, LCD_I2C_ADDRESS, cols=16, rows=2)
    button = Button(BUTTON_PIN, bounce_time=0.1)

    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration(main={"format": "RGB888"}))
    picam2.start()
    time.sleep(1)  # let auto-exposure settle

    try:
        show_idle(lcd)
        while True:
            button.wait_for_press()
            show_processing(lcd)

            class_index, confidence = poll_and_vote(interpreter, picam2)

            try:
                show_result(lcd, class_index, confidence)
            except Exception as exc:
                print(f"LCD write failed: {exc}")

            time.sleep(RESULT_HOLD_S)
            show_idle(lcd)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        lcd.close(clear=True)


if __name__ == "__main__":
    main()
