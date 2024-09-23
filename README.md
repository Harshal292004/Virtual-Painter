# Hand Gesture Detection and Tracking

This project implements a real-time hand gesture detection and tracking system using OpenCV and MediaPipe.

## Features

- Real-time hand detection
- Finger position tracking
- Gesture recognition (fingers up/down)
- FPS display

## Tech Stack

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy

## Installation

1. Clone the repository:git clone https://github.com/yourusername/your-repo-name.git
2. Install the required packages:requirements.txt

## Usage

Run the main script:

This will open your webcam feed and start detecting hand gestures in real-time.

## How it works

The `handDetector` class uses MediaPipe's hand tracking solution to detect and track hands in the video feed. It provides methods to:

- Detect hands in an image
- Find the position of hand landmarks
- Determine which fingers are raised

The main loop captures video from the webcam, processes each frame to detect hands and fingers, and displays the results in real-time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
