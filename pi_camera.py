#!/usr/bin/env python3

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io
import time
import threading

PORT = 8000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAMERATE = 24
QUALITY = 85

app = Flask(__name__)
picam2 = Picamera2()

latest_frame_buffer = None
frame_lock = threading.Lock()

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def capture_frames_continuously():
    """
    Captures frames from the camera and updates the global buffer.
    This function runs in a separate thread.
    """
    global latest_frame_buffer
    print("Starting continuous frame capture thread...")
    try:
        video_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            lores={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "YUV420"},
            controls={"FrameRate": FRAMERATE}
        )
        print("  Thread: Configuring camera...")
        picam2.configure(video_config)

        output = StreamingOutput()
        encoder = JpegEncoder(q=QUALITY)

        print("  Thread: Starting camera recording...")
        picam2.start_recording(encoder, FileOutput(output))

        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame_data = output.frame
                with frame_lock:
                    latest_frame_buffer = frame_data
        except Exception as e:
            print(f"Error in capture loop: {e}")
        finally:
            print("  Thread: Stopping camera recording...")
            picam2.stop_recording()
            print("Frame capture thread finished.")
    except Exception as e:
        print(f"Error in capture thread setup: {e}")


def generate_camera_stream():
    """Generator function that yields JPEG frames for the HTTP response."""
    global latest_frame_buffer
    print("Client connected to camera stream.")
    try:
        while True:
            with frame_lock:
                frame_to_send = latest_frame_buffer

            if frame_to_send is None:
                time.sleep(0.01)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            time.sleep(1/FRAMERATE * 0.8)
    except GeneratorExit:
        print("Client disconnected from camera stream.")
    except Exception as e:
        print(f"Error in stream generator: {e}")

@app.route('/')
def index():
    """Serves the main HTML page with the embedded MJPEG stream."""
    html_content = """
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f0f0f0; text-align: center; }
            h1 { color: #333; }
            img {
                display: block;
                margin-left: auto;
                margin-right: auto;
                border: 3px solid #333;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi Camera Module 3 Stream</h1>
        <img src="{{ url_for('video_feed') }}" width="{{ frame_width }}" height="{{ frame_height }}">
        <p>Refresh the page if the stream doesn't start immediately.</p>
    </body>
    </html>
    """
    return render_template_string(html_content, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)

@app.route('/video_feed')
def video_feed():
    """Route that serves the MJPEG stream."""
    return Response(generate_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        print("Program starting...")

        capture_thread = threading.Thread(target=capture_frames_continuously, daemon=True)
        capture_thread.start()

        time.sleep(2)

        print(f"Starting Flask web server on http://0.0.0.0:{PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        if picam2.started:
            print("Ensuring Picamera2 is stopped in main finally block (if it was started globally)...")
        print("Program ended.")