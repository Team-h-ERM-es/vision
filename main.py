import cv2
from flask import Flask, Response, render_template_string
import threading
from ultralytics import YOLO
import time
import json

model = YOLO()

app = Flask(__name__)

frame = None
detections = []
lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            display: flex;
            flex-direction: row;
            gap: 20px;
            width: 100%;
            max-width: 1200px;
        }
        .video-container {
            flex: 2;
        }
        .info-container {
            flex: 1;
            overflow-y: auto;
            height: 480px;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        h2 {
            margin-top: 0;
        }
        .detection-item {
            margin-bottom: 10px;
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
    </style>
</head>
<body>
    <h1>Object Detection Stream</h1>
    <div class="container">
        <div class="video-container">
            <img src="/video_feed" width="640" height="480">
        </div>
        <div class="info-container">
            <h2>Detection Information</h2>
            <div id="detections"></div>
        </div>
    </div>

    <script>
        // Function to fetch and update detections
        async function updateDetections() {
            try {
                const response = await fetch('/detections');
                const data = await response.json();
                
                const detectionsDiv = document.getElementById('detections');
                detectionsDiv.innerHTML = '';
                
                data.forEach((detection, index) => {
                    const div = document.createElement('div');
                    div.className = 'detection-item';
                    div.innerHTML = `
                        <strong>${detection.class_name}</strong> (Conf: ${detection.confidence.toFixed(2)})<br>
                        Box: [${detection.bbox.map(v => Math.round(v)).join(', ')}]<br>
                        Center: (${Math.round(detection.center_x)}, ${Math.round(detection.center_y)})
                    `;
                    detectionsDiv.appendChild(div);
                });
            } catch (error) {
                console.error('Error fetching detections:', error);
            }
            
            // Update every 100ms
            setTimeout(updateDetections, 100);
        }
        
        // Start updating detections
        updateDetections();
    </script>
</body>
</html>
"""

def capture_video():
    global frame, detections
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        results = model(img)
        result = results[0]
        
        current_detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = result.names[cls]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            current_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': cls,
                'class_name': class_name,
                'center_x': center_x,
                'center_y': center_y
            })
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        with lock:
            frame = img.copy()
            detections = current_detections
            
    cap.release()

def generate_frames():
    global frame
    
    while True:
        if frame is None:
            time.sleep(0.1)
            continue
            
        with lock:
            img = frame.copy()
            
        _, buffer = cv2.imencode('.jpg', img)
        jpg_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    with lock:
        return json.dumps(detections)

if __name__ == '__main__':
    threading.Thread(target=capture_video, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5001, debug=False)