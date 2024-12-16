import cv2
import torch
import time
import numpy as np
import csv
import os
from datetime import datetime
import logging
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path



# Configure logging
logging.basicConfig(
    filename='warnings.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)
logging.captureWarnings(True)
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.autocast`.*", category=FutureWarning)

weights_path = Path('weights') / 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path.resolve()), trust_repo=True)


# Camera Calibration Values
mtx = [[469.14058291, 0., 328.09378798],
       [0., 465.55755617, 279.88263823],
       [0., 0., 1.]]
dist = [0.04234017, -0.07239547, 0.00408545, 0.00331313, -0.08531045]


# # Camera Calibration Values
# mtx = [[574.2511204, 0., 303.04699623],
#        [0., 575.89185138, 273.67241784],
#        [0., 0., 1.]]
# dist = [0.13034276, -0.32444014, -0.00167153, -0.00040383, 0.06335327]



# Distances and PPI values for dynamic calculation
distances = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
ppi_values = [204.594450, 151.082759, 116.443978, 97.255962, 84.082078, 72.182278, 64.793880, 58.000086, 52.188232,
              48.701186, 44.617042, 41.867441, 38.416721, 36.380369, 33.884191, 32.237924, 30.405338, 29.000043,
              27.619212, 26.422363, 25.095238, 24.090909, 23.095680, 22.003005, 21.382510, 20.827222, 19.960358, 19.132213]

# TRACKED_OBJECTS Structure: {obj_id: (bbox, frames_seen, first_saved)}
TRACKED_OBJECTS = {}
NEXT_OBJECT_ID = 0
MAX_MISSED_FRAMES = 3
IOU_THRESHOLD = 0.5




# Helper functions
def undistort_frame(frame, mtx, dist):
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx), np.array(dist), (w, h), 1, (w, h))
    return cv2.undistort(frame, np.array(mtx), np.array(dist), None, new_camera_mtx)


def dynamic_ppi(distance):
    # Ensure distance is within the range of predefined distances
    if distance < min(distances) or distance > max(distances):
        print(f"Warning: Distance {distance} is out of range. Using closest valid PPI.")
        distance = max(min(distances), min(max(distances), distance))
    
    # Interpolate PPI values
    interpolate_ppi = interp1d(distances, ppi_values, kind='linear', fill_value="extrapolate")
    ppi = interpolate_ppi(distance)
    return ppi


def calculate_real_world_dimensions(bbox_width_pixels, bbox_height_pixels, ppi):
    real_width = bbox_width_pixels / ppi
    real_height = bbox_height_pixels / ppi
    return real_width, real_height

def estimate_distance(focal_length, real_width, bbox_width_pixels):
    return (focal_length * real_width) / bbox_width_pixels

def process_frame(frame, model, real_object_width, focal_length):
    # Undistort the frame
    undistorted = undistort_frame(frame, mtx, dist)

    # Detect objects with YOLOv5
    results = model(undistorted)
    detections = results.xyxy[0].cpu().numpy()

    # Process the first detected object (if any)
    if len(detections) > 0:
        bbox = detections[0][:4]  # Extract bounding box
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integer coordinates

        # Calculate bbox dimensions
        bbox_width_pixels = x2 - x1
        bbox_height_pixels = y2 - y1

        # Add center marker for debugging
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Estimate distance
        distance_in_inches = estimate_distance(focal_length, real_object_width, bbox_width_pixels)

        # Calculate PPI
        ppi = dynamic_ppi(distance_in_inches)
        if ppi == 0:
            print(f"Error: Calculated PPI is zero for distance {distance_in_inches}. Skipping this detection.")
            return None, None, None, None

        # Calculate real-world dimensions
        real_width, real_height = calculate_real_world_dimensions(bbox_width_pixels, bbox_height_pixels, ppi)

        return real_width, real_height, distance_in_inches, bbox

    return None, None, None, None

def track_objects(detections):
    global TRACKED_OBJECTS, NEXT_OBJECT_ID

    matched_objects = {}
    for det in detections:
        bbox = det[:4]
        best_match_id = None
        highest_iou = IOU_THRESHOLD

        # Match detection to existing tracked objects
        for obj_id, (tracked_bbox, frames_seen, first_saved) in TRACKED_OBJECTS.items():
            iou = calculate_iou(tracked_bbox, bbox)
            if iou > highest_iou:
                highest_iou = iou
                best_match_id = obj_id

        if best_match_id is not None:
            matched_objects[best_match_id] = (bbox, TRACKED_OBJECTS[best_match_id][1] + 1, TRACKED_OBJECTS[best_match_id][2])
        else:
            # Create a new object ID if no match is found
            matched_objects[NEXT_OBJECT_ID] = (bbox, 1, False)  # Initialize first_saved as False
            NEXT_OBJECT_ID += 1

    # Remove objects not matched in the current frame
    TRACKED_OBJECTS = {obj_id: matched_objects[obj_id] for obj_id in matched_objects}

    return TRACKED_OBJECTS




def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def run_detection(duration, stop_event, session_folder=None, video_path=None):
    """Run object detection for live feed or video."""
    global TRACKED_OBJECTS, NEXT_OBJECT_ID

    if session_folder is None:
        session_folder = create_session_folder()

    csv_file_path = session_folder / 'fish_weight_data.csv'
    write_csv_headers(csv_file_path)

    # Open video capture (camera or video file)
    cap = cv2.VideoCapture(str(video_path) if video_path else 0)
    focal_length = 469.14  # From calibration
    real_object_width = 8.0

    start_time = time.time()  # Record the session start time

    try:
        while cap.isOpened():
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if stop_event.is_set():  # Stop detection if the event is triggered
                print("Stop event triggered. Ending detection.")
                break
            if duration > 0 and elapsed_time >= duration:  # Stop if duration is exceeded
                print(f"Detection ended after {elapsed_time:.2f}s (duration = {duration}s).")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            # Process the frame and handle detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            if len(detections) > 0:
                tracked_objects = track_objects(detections)

                for obj_id, (tracked_bbox, _, _) in tracked_objects.items():
                    x1, y1, x2, y2 = map(int, tracked_bbox)
                    bbox_width_pixels = x2 - x1
                    bbox_height_pixels = y2 - y1

                    distance = estimate_distance(focal_length, real_object_width, bbox_width_pixels)
                    ppi = dynamic_ppi(distance)

                    if ppi > 0:
                        real_width, real_height = calculate_real_world_dimensions(bbox_width_pixels, bbox_height_pixels, ppi)

                        # Annotate bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"Distance: {distance:.2f} in", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        save_detection_data(csv_file_path, frame, obj_id, tracked_bbox, real_width, real_height, distance)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'

    finally:
        TRACKED_OBJECTS.clear()
        NEXT_OBJECT_ID = 0
        calculate_averages(csv_file_path)
        cap.release()




def calculate_averages(csv_file_path):
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            data = [row for row in reader if row]  # Collect all rows

        if not data:
            print("No data available to calculate averages.")
            return

        widths = [float(row[2]) for row in data]  # Extract widths
        heights = [float(row[3]) for row in data]  # Extract heights
        weights = [float(row[4]) for row in data]  # Extract weights

        avg_width = round(np.mean(widths), 3)
        avg_height = round(np.mean(heights), 3)
        avg_weight = round(np.mean(weights), 3)

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Average", "", avg_width, avg_height, avg_weight, ""])

        print(f"Averages added to CSV: Width={avg_width:.3f}, Height={avg_height:.3f}, Weight={avg_weight:.3f}")
    except Exception as e:
        print(f"Error calculating averages: {e}")



def write_csv_headers(csv_file_path):
    with csv_file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "ID", "Width (in)", "Height (in)", "Weight (g)", "Distance (in)"])


def save_detection_data(csv_file_path, frame, obj_id, bbox, real_width, real_height, distance):
    global TRACKED_OBJECTS

    if obj_id not in TRACKED_OBJECTS:
        print(f"[DEBUG] Object ID {obj_id} not found in TRACKED_OBJECTS at the time of saving.")
        return

    print(f"[DEBUG] Saving data for Object ID {obj_id}.")
    # Proceed with saving logic as updated above


    x1, y1, x2, y2 = map(int, bbox)
    weight = calculate_weight(real_width, real_height)

    # Round values to three decimal places
    real_width = round(real_width, 3)
    real_height = round(real_height, 3)
    weight = round(weight, 3)
    distance = round(distance, 3)

    # Save only on the first frame
    if not TRACKED_OBJECTS[obj_id][2]:  # Use the first_saved flag
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_folder = Path(csv_file_path).parent
        image_filename = f"fish_{obj_id}_{timestamp.replace(':', '-')}.jpg"
        image_path = session_folder / image_filename
        cv2.imwrite(image_path, frame)

        # Save data to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, obj_id, real_width, real_height, weight, distance])

        # Mark the object as saved
        TRACKED_OBJECTS[obj_id] = (TRACKED_OBJECTS[obj_id][0], TRACKED_OBJECTS[obj_id][1], True)



def get_averages_from_csv(csv_file_path):
    try:
        lengths, widths, weights = [], [], []
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Timestamp'] != "Average":  # Skip the averages row
                    try:
                        lengths.append(float(row["Width (in)"]))
                        widths.append(float(row["Height (in)"]))
                        weights.append(float(row["Weight (g)"]))
                    except ValueError:
                        print(f"Skipping invalid row: {row}")

        if lengths and widths and weights:
            return {
                'average_length': round(sum(lengths) / len(lengths), 3),
                'average_width': round(sum(widths) / len(widths), 3),
                'average_weight': round(sum(weights) / len(weights), 3),
            }
        else:
            print(f"No valid data found in CSV: {csv_file_path}")
            return {
                'average_length': 0,
                'average_width': 0,
                'average_weight': 0,
            }
    except Exception as e:
        print(f"Error fetching averages from CSV: {e}")
        return {
            'average_length': 0,
            'average_width': 0,
            'average_weight': 0,
        }




def calculate_weight(length, width):
    """
    Calculate the weight of the fish based on length and width (in inches).
    The formula calculates weight in pounds and converts it to grams.
    Correct formula: [Length × ((Width × 2) × 2)] / 800
    """
    weight_in_pounds = (length * ((width * 2) * (width * 2))) / 712.57  # Updated formula
    weight_in_grams = weight_in_pounds * 453.592  # Convert pounds to grams
    return round(weight_in_grams, 3)






def create_session_folder():
    base_folder = Path('detected_fish_data')
    base_folder.mkdir(exist_ok=True)
    base_folder.mkdir(exist_ok=True)

    session_folder = base_folder / f"Detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        session_folder.mkdir(parents=True, exist_ok=True)
        return session_folder
    except Exception as e:
        print(f"Failed to create session folder: {e}")
        return None

if __name__ == '__main__':
    session_folder = create_session_folder()
    csv_file_path = session_folder / 'fish_weight_data.csv'
    with csv_file_path.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "ID", "Width (in)", "Height (in)", "Weight (g)", "Distance (in)"])

    cap = cv2.VideoCapture(0)
    focal_length = 469.14  # From calibration
    real_object_width = 8.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for detections
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) > 0:
            tracked_objects = track_objects(detections)

            # Process tracked objects
            for obj_id, (tracked_bbox, _, _) in tracked_objects.items():
                x1, y1, x2, y2 = map(int, tracked_bbox)
                bbox_width_pixels = x2 - x1
                bbox_height_pixels = y2 - y1

                # Estimate distance and calculate dimensions
                distance = estimate_distance(focal_length, real_object_width, bbox_width_pixels)
                ppi = dynamic_ppi(distance)
                if ppi > 0:
                    real_width, real_height = calculate_real_world_dimensions(bbox_width_pixels, bbox_height_pixels, ppi)

                    # Annotate bounding box, ID, and distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Distance: {distance:.2f} in", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Save detection data
                    save_detection_data(csv_file_path, frame, obj_id, tracked_bbox, real_width, real_height, distance)
                else:
                    print(f"Warning: PPI is zero for object ID {obj_id}. Skipping this detection.")

        cv2.imshow('Fish Weight Calculator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and append averages to the CSV
    calculate_averages(csv_file_path)

