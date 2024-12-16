from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
import cv2
from threading import Thread, Event, Lock
import time
from datetime import datetime
import os
import zipfile
import shutil
import fish_weight_calculator
from pathlib import Path


app = Flask(__name__)
current_session_folder = None
thread = None
stop_event = Event()
average_data = {}
start_time = None
remaining_time = None
session_active = False
lock = Lock()  # Lock to manage access to frames in case of concurrent access




# Video directory
VIDEOS_FOLDER = Path.cwd() / "videos_for_detection"
VIDEOS_FOLDER.mkdir(exist_ok=True)





# Route to serve video files
@app.route('/static/videos_for_detection/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEOS_FOLDER, filename)

# Main page route
@app.route('/')
def index():
    global remaining_time, start_time, session_active, stop_event, current_session_folder

    # Check if session is active
    if session_active:
        # Stop the current session and save state
        elapsed = time.time() - start_time if start_time else 0
        remaining_time = max(0, int(remaining_time - elapsed))
        stop_event.set()
        session_active = False

        # Save data and stop detection
        csv_file_path = current_session_folder / 'fish_weight_data.csv'
        fish_weight_calculator.calculate_averages(csv_file_path)
        print("Detection stopped. Data saved.")

    # Render the index page if no session is active
    return render_template('index.html', session_active=session_active)



# Route for video list page
@app.route('/videos')
def videos_page():
    # Include additional video formats
    allowed_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    videos = [f for f in os.listdir(VIDEOS_FOLDER) if any(f.lower().endswith(ext) for ext in allowed_extensions)]
    return render_template('videos.html', videos=videos)


# Route to handle video uploads
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected file", 400

    # Extend accepted formats
    allowed_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    if video and any(video.filename.lower().endswith(ext) for ext in allowed_extensions):
        video.save(VIDEOS_FOLDER / video.filename)
        return redirect(url_for('videos_page'))
    return "Invalid file format", 400


@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    video_path = VIDEOS_FOLDER / video_name
    if not video_path.exists():
        return "Error: Video file not found.", 404

    stop_event.clear()
    return Response(fish_weight_calculator.run_detection(-1, stop_event, video_path=video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Start video detection
@app.route('/start_video_detection/<video_name>', methods=['GET'])
def start_video_detection(video_name):
    global thread, stop_event, session_active, start_time, remaining_time
    video_path = os.path.join(VIDEOS_FOLDER, video_name)

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(frame_count // fps)
    cap.release()

    stop_event.clear()
    start_time = time.time()
    remaining_time = duration
    session_active = True

    if thread is None or not thread.is_alive():
        thread = Thread(target=run_detection_with_redirect, args=(duration, video_path))
        thread.daemon = True
        thread.start()

    return redirect(url_for('start_page', duration=duration, video_name=video_name))

# Run detection and redirect on completion
def run_detection_with_redirect(duration, video_path=None):
    global average_data, session_active, remaining_time, current_session_folder
    try:
        session_active = True
        remaining_time = duration

        # Ensure the session folder is initialized
        if current_session_folder is None:
            current_session_folder = fish_weight_calculator.create_session_folder()

        # Pass current_session_folder and video_path to run_detection
        generator = fish_weight_calculator.run_detection(duration, stop_event, session_folder=current_session_folder, video_path=video_path)
        for _ in generator:
            if stop_event.is_set():
                break

        # Calculate averages once detection ends
        csv_file_path = current_session_folder / 'fish_weight_data.csv'        
        average_data = fish_weight_calculator.calculate_averages(csv_file_path)
        print("Detection completed and averages saved.")

    except Exception as e:
        print(f"Error in run_detection_with_redirect: {e}")
    finally:
        session_active = False



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


# Video stream generator for live feed from fish_weight_calculator
@app.route('/live_video_feed')
def live_video_feed():
    duration = int(request.args.get('duration', -1))  # Default to -1 if not provided
    stop_event.clear()
    return Response(fish_weight_calculator.run_detection(duration, stop_event), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/start', methods=['POST'])
def start():
    global thread, stop_event, start_time, remaining_time, session_active
    try:
        if session_active:
            elapsed = time.time() - start_time if start_time else 0
            remaining_time = max(0, int(remaining_time - elapsed))
            return redirect(url_for('start_page', duration=remaining_time))

        # Parse and validate duration
        duration = int(request.form.get('duration', 0))
        if duration <= 0:
            return "Invalid duration. Must be greater than zero.", 400

        stop_event.clear()  # Reset stop signal
        start_time = time.time()
        remaining_time = duration
        session_active = True

        # Start detection thread
        if thread is None or not thread.is_alive():
            thread = Thread(target=fish_weight_calculator.run_detection, args=(duration, stop_event))
            thread.daemon = True
            thread.start()

        return redirect(url_for('start_page', duration=duration))
    except Exception as e:
        print(f"Error starting detection: {e}")
        return "Internal Server Error", 500



# Route for detection in progress page
@app.route('/start_page')
def start_page():
    global remaining_time, stop_event

    if session_active:
        elapsed = time.time() - start_time if start_time else 0
        remaining_time = max(0, int(remaining_time - elapsed))
    else:
        remaining_time = 0  # Reset if no session is active

    video_name = request.args.get('video_name')  # Get video_name if provided
    return render_template('start.html', duration=remaining_time, video_name=video_name)



# Stop detection route
@app.route('/stop')
def stop():
    global stop_event, session_active
    stop_event.set()  # Signal to stop detection
    session_active = False
    return redirect(url_for('end'))


# Display results after detection ends
@app.route('/end')
def end():
    global current_session_folder

    base_folder = Path('detected_fish_data')
    average_data = {}

    # Find the latest session folder if current_session_folder is None
    if not current_session_folder:
        session_folders = [f for f in base_folder.iterdir() if f.is_dir()]
        if session_folders:
            # Sort session folders by modification time and pick the latest
            current_session_folder = max(session_folders, key=lambda f: f.stat().st_mtime)

    # Ensure current_session_folder is a Path object
    current_session_folder = Path(current_session_folder)

    # Fetch averages from the CSV file
    if current_session_folder:
        csv_file_path = current_session_folder / 'fish_weight_data.csv'
        if csv_file_path.exists():
            average_data = fish_weight_calculator.get_averages_from_csv(csv_file_path)
        else:
            print(f"CSV file not found in session folder: {csv_file_path}")
    else:
        print("No session folder found to fetch averages.")

    # Reset current_session_folder after fetching data
    current_session_folder = None
    return render_template('end.html', average_data=average_data)




# Route to list all session data folders
@app.route('/data')
def list_data():
    base_folder = Path('detected_fish_data')
    sessions = [session for session in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, session))]
    return render_template('data.html', sessions=sessions)


# Route to download a session folder as a zip
@app.route('/download_session/<session>')
def download_session(session):
    base_folder = Path('detected_fish_data')
    session_path = base_folder / session
    if not os.path.isdir(session_path):
        return f"Session folder '{session}' does not exist.", 404

    zip_filename = f"{session}.zip"
    zip_path = base_folder / zip_filename

    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(session_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(str(file_path), os.path.relpath(file_path, str(base_folder)))

        if not os.path.exists(zip_path):
            return "ZIP file could not be created.", 500

        return send_from_directory(base_folder, zip_filename, as_attachment=True)

    except Exception as e:
        print(f"Error creating or sending ZIP file: {e}")
        return f"An error occurred while creating or sending the ZIP file for session '{session}'", 500

# Clean up zip file after download
@app.after_request
def cleanup_zip(response):
    for file in os.listdir():
        if file.endswith(".zip"):
            os.remove(file)
    return response

@app.route('/delete_session/<session>', methods=['POST'])
def delete_session(session):
    base_folder = Path('detected_fish_data')
    session_path = base_folder / session
    zip_path = base_folder / f"{session}.zip"

    if not session_path.is_dir():
        return jsonify({"success": False, "error": f"Session folder '{session}' does not exist."}), 404

    try:
        # Delete the session folder
        shutil.rmtree(str(session_path))

        # Delete the corresponding ZIP file if it exists
        if zip_path.is_file():
            zip_path.unlink()

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error deleting session folder '{session}': {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    
@app.route('/delete_video/<video>', methods=['POST'])
def delete_video(video):
    video_path = VIDEOS_FOLDER / video  # Use Path object for platform-independent path handling

    if not video_path.is_file():  # Correctly check if it's a file
        return jsonify({"success": False, "error": f"Video '{video}' does not exist."}), 404

    try:
        video_path.unlink()  # Delete the video file
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error deleting video '{video}': {e}")
        return jsonify({"success": False, "error": str(e)}), 500



def log_detection_status():
    while True:
        if session_active:
            print("Detection ongoing")
        time.sleep(1)  # Log every 5 seconds


if __name__ == '__main__':
    debug_thread = Thread(target=log_detection_status)
    debug_thread.daemon = True
    debug_thread.start()

    app.run(host='0.0.0.0', port=5001, debug=True)