# Monkey Gesture Control

Real-time webcam app that matches your live pose to monkey meme poses and shows the matched image beside your camera feed.

## Features

- Pose matching for 5 monkey images:
  - `Monkey1`: both hands on head + open mouth
  - `Monkey2`: dab pose
  - `Monkey3`: thumbs up
  - `Monkey4`: index finger near mouth
  - `Monkey5`: peace sign + smile
- Safety action:
  - Middle finger (any hand) held for 2 seconds triggers shutdown
- Side-by-side output:
  - Camera feed on the left, matched monkey image on the right

## Project Structure

```text
Monkey/
├── gesture_control.py
├── requirements.txt
├── README.md
└── assets/
    ├── Monkey1.jpg
    ├── Monkey2.jpg
    ├── Monkey3.jpg
    ├── Monkey4.jpg
    └── Monkey5.jpg
```

## Requirements

- Python 3.11
- Webcam

Install dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Run

```powershell
py -3.11 gesture_control.py
```

Press `Q` to quit.

## Notes

- If any image is missing, a placeholder is used automatically.
- `SHUTDOWN_CONFIRM = False` means shutdown executes immediately after gesture hold.
- Set `SHUTDOWN_CONFIRM = True` in `gesture_control.py` if you want a terminal confirmation before shutdown.
