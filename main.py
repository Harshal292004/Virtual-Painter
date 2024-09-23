import cv2
import numpy as np
import time 
import os 
from hand_detector import HandDetector
import math
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def send_to_ai(img_canvas, fingers):
    if tuple(fingers) == (0, 1, 1, 1, 1):
        pil_image = Image.fromarray(img_canvas)
        try:
            response = model.generate_content(['Solve this Math Problem', pil_image])
            print(response.text.encode('utf-8', 'ignore').decode('utf-8'))
        except Exception as e:
            print(f"Error sending to AI: {e}")

def is_finger_raised(finger_tip_y, finger_base_y, threshold, img):
    height_difference = finger_base_y - finger_tip_y
    is_raised = height_difference > threshold
    cv2.putText(img, f'Distance:{int(height_difference)} {is_raised}', (640, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    return is_raised

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    header_link = 'HEADER'
    header_list = os.listdir('HEADER')

    color_list = [
        (210, 50, 230),  # Pink brush
        (255, 0, 0),  # Blue brush
        (0, 252, 124),  # Green brush
        (0, 0, 0)  # eraser
    ]

    pTime = 0
    detector = HandDetector(detectionCon=0.7, trackCon=0.7, maxHands=1)
    xp, yp = 0, 0

    brush_thickness = 5
    eraser_thickness = 50

    # Loading header and wipe utility
    overlay_list = []
    for img_file in header_list:
        image = cv2.imread(os.path.join(header_link, img_file))
        if image is not None:
            overlay_list.append(image)
        else:
            print(f"Failed to load image: {img_file}")
    selected_color = color_list[0]
    header = overlay_list[0]

    wipe = cv2.imread('wipe.webp')
    wipe = cv2.resize(wipe, (170, 170))

    # Creating a canvas for drawing
    img_canvas = np.zeros((720, 1280, 3), np.uint8)

    while True:
        # Capture and process image
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        img = cv2.flip(img, 1)

        # Add header and wipe utility
        img[0:125, 0:1280] = header
        img[550:720, 0:170] = wipe

        # Detect hand and fingers
        detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)
        fingers = detector.fingersUp(lm_list)

        if len(lm_list) != 0:
            # Get finger positions
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            # Selection Mode
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                cv2.rectangle(img, (x1 - 15, y1 - 25), (x2 + 15, y2 + 25), selected_color, cv2.FILLED)
                print('Selection Mode')
                if y1 < 125:
                    # Handle color selection
                    if 250 < x1 < 450:
                        header, selected_color = overlay_list[0], color_list[0]
                    elif 550 < x1 < 750:
                        header, selected_color = overlay_list[1], color_list[1]
                    elif 800 < x1 < 950:
                        header, selected_color = overlay_list[2], color_list[2]
                    elif 1050 < x1 < 1200:
                        header, selected_color = overlay_list[3], color_list[3]
                elif 550 < y1 < 700:
                    # Wipe the image
                    img_canvas = np.zeros((720, 1280, 3), np.uint8)

            # Drawing Mode
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 5, selected_color, 3)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if selected_color == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), selected_color, eraser_thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), selected_color, eraser_thickness)
                else:
                    if not is_finger_raised(lm_list[8][2], lm_list[5][2], 145, img):
                        cv2.line(img, (xp, yp), (x1, y1), selected_color, brush_thickness)
                        cv2.line(img_canvas, (xp, yp), (x1, y1), selected_color, brush_thickness)
                xp, yp = x1, y1

        # Merge canvas with camera feed
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        if fingers:
            send_to_ai(img_canvas, fingers)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow('Drawing Application', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()