import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io
import uuid
import json
from datetime import datetime
import shutil

# Import the DeepfakeDetector class from your existing file
from deepfake_detector import DeepfakeDetector, IMG_SIZE

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create symbolic link to performance_metrics folder in static directory to make plots accessible
METRICS_DIR = 'performance_metrics'
PLOTS_DIR = os.path.join(METRICS_DIR, 'evaluation', 'plots')
STATIC_METRICS_DIR = os.path.join('static', 'metrics')

# Create metrics directory in static folder if it doesn't exist
os.makedirs(STATIC_METRICS_DIR, exist_ok=True)

# Check for GPU availability
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")

# Global variables
model = None
default_threshold = 0.9
detection_history = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def load_model():
    try:
        model = DeepfakeDetector().to(device)
        # Try to load the final model first, fall back to best model if needed
        try:
            model.load_state_dict(torch.load('performance_metrics/best_model.pth', 
                                           map_location=device))
            print("Loaded final model")
        except:
            model.load_state_dict(torch.load('performance_metrics/best_model.pth', 
                                           map_location=device))
            print("Loaded best model")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(image_array, threshold=0.5):
    """Predict if an image is deepfake and return results"""
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return {"error": "Failed to load model"}
    
    try:
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
        else:
            return {"error": "Invalid image format"}
            
        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image_array).unsqueeze(0).to(device)
        
        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output.squeeze()).item()
        
        # Classify based on threshold
        is_deepfake = prob > threshold
        
        result = {
            "probability": float(prob),
            "threshold": float(threshold),
            "is_deepfake": bool(is_deepfake),
            "label": "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC IMAGE"
        }
        
        return result
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"error": str(e)}

def analyze_video(video_path, threshold=0.5, max_frames=100):
    """Analyze a video file for deepfakes"""
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            return {"error": "Failed to load model"}
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        # Initialize counters
        total_frames = 0
        deepfake_frames = 0
        real_frames = 0
        sampled_frames = 0
        frame_results = []
        
        # Get total frame count
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frame_count / fps if fps > 0 else 0
        
        # Calculate frame interval to sample approximately max_frames
        frame_interval = max(1, int(total_frame_count / max_frames))
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if total_frames % frame_interval == 0:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = predict_image(frame_rgb, threshold)
                prob = result.get("probability", 0)
                
                # Store timestamp
                timestamp = total_frames / fps if fps > 0 else 0
                
                # Update counters
                sampled_frames += 1
                if prob > threshold:
                    deepfake_frames += 1
                else:
                    real_frames += 1
                
                # Save frame result
                frame_results.append({
                    "frame_number": total_frames,
                    "timestamp": timestamp,
                    "probability": prob,
                    "is_deepfake": prob > threshold
                })
            
            total_frames += 1
        
        # Release video capture
        cap.release()
        
        # Calculate final result
        deepfake_percentage = 0
        if sampled_frames > 0:
            deepfake_percentage = (deepfake_frames / sampled_frames) * 100
        
        # Explicitly set is_deepfake based on our rules
        # Rule 1: Default is based on percentage (>50% means deepfake)
        is_deepfake = deepfake_percentage > 50
        
        # Rule 2: If there are at least 4 authentic frames, the video is authentic
        # regardless of the deepfake percentage
        if real_frames >= 25:
            is_deepfake = False
            print(f"Video classified as AUTHENTIC because it has {real_frames} authentic frames (≥4)")
        else:
            if is_deepfake:
                print(f"Video classified as DEEPFAKE: {deepfake_percentage:.1f}% of frames are manipulated")
            else:
                print(f"Video classified as AUTHENTIC: {100-deepfake_percentage:.1f}% of frames are authentic")
        
        result = {
            "deepfake_percentage": deepfake_percentage,
            "is_deepfake": is_deepfake,
            "label": "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC VIDEO",
            "total_frames": total_frame_count,
            "sampled_frames": sampled_frames,
            "deepfake_frames": deepfake_frames,
            "real_frames": real_frames,
            "duration": duration,
            "threshold": threshold,
            "frame_results": frame_results
        }
        
        return result
    
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {"error": str(e)}

def add_to_history(file_type, filename, result):
    """Add a detection result to history"""
    global detection_history
    
    # Create history entry
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_type": file_type,
        "filename": filename,
        "result": result
    }
    
    # Add to history (limit to latest 20 entries)
    detection_history.insert(0, entry)
    detection_history = detection_history[:20]
    
    return entry["id"]

@app.route('/')
def index():
    # Load training stats if available
    training_stats = {}
    try:
        with open('training_stats.json', 'r') as f:
            training_stats = json.load(f)
    except:
        pass
    
    # Check if metrics are available
    metrics_path = 'performance_metrics/evaluation/metrics.json'
    has_metrics = os.path.exists(metrics_path)
    metrics = {}
    
    if has_metrics:
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            pass
    
    return render_template('index.html', 
                           device=str(device),
                           gpu_available=gpu_available,
                           threshold=default_threshold,
                           history=detection_history,
                           training_stats=training_stats,
                           has_metrics=has_metrics,
                           metrics=metrics)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    threshold = float(request.form.get('threshold', default_threshold))
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process based on file type
        if is_video_file(filename):
            # Video processing
            result = analyze_video(file_path, threshold)
            if "error" not in result:
                # Save unique thumbnail
                thumbnail_path = os.path.splitext(file_path)[0] + "_thumbnail.jpg"
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(thumbnail_path, frame)
                cap.release()
                
                # Add to history
                history_id = add_to_history("video", filename, result)
                result["history_id"] = history_id
                result["thumbnail"] = os.path.basename(thumbnail_path)
                result["file_path"] = os.path.basename(file_path)
                
            return render_template('result.html', 
                                  result=result, 
                                  file_type="video",
                                  filename=filename)
        else:
            # Image processing
            try:
                # Read the image
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                result = predict_image(image_rgb, threshold)
                
                if "error" not in result:
                    # Add to history
                    history_id = add_to_history("image", filename, result)
                    result["history_id"] = history_id
                    result["file_path"] = os.path.basename(file_path)
                
                return render_template('result.html', 
                                      result=result, 
                                      file_type="image",
                                      filename=filename)
            
            except Exception as e:
                flash(f"Error processing file: {str(e)}")
                return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/webcam', methods=['POST'])
def webcam_detect():
    try:
        # Get image data from the request
        image_data = request.json.get('image')
        threshold = float(request.json.get('threshold', default_threshold))
        
        # Convert base64 to image
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_array = np.array(image)
        
        # Save the image
        filename = f"webcam_{uuid.uuid4()}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)
        
        # Process the image
        result = predict_image(image_array, threshold)
        
        if "error" not in result:
            # Add to history
            history_id = add_to_history("webcam", filename, result)
            result["history_id"] = history_id
            result["file_path"] = filename
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/history/<history_id>')
def view_history_item(history_id):
    # Find the history item by ID
    for item in detection_history:
        if item["id"] == history_id:
            return render_template('result.html', 
                                  result=item["result"], 
                                  file_type=item["file_type"],
                                  filename=item["filename"],
                                  from_history=True)
    
    flash("History item not found")
    return redirect(url_for('index'))

@app.route('/about')
def about():
    # Get training stats if available
    training_stats = {}
    try:
        with open('training_stats.json', 'r') as f:
            training_stats = json.load(f)
    except:
        pass
    
    return render_template('about.html', training_stats=training_stats, 
                          has_metrics=os.path.exists('performance_metrics/evaluation/metrics.json'))

@app.route('/performance')
def performance():
    """Display model performance metrics and visualizations"""
    # Get metrics data if available
    metrics = {}
    eval_path = 'performance_metrics/evaluation'
    metrics_path = os.path.join(eval_path, 'metrics.json')
    
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
                # Validate expected metrics structure
                if 'classification_report' in metrics:
                    # Ensure classification_report has expected keys
                    class_report = metrics['classification_report']
                    class_keys = list(class_report.keys())
                    
                    # Make sure we have at least '0'/'1' or macro/weighted keys
                    has_valid_keys = False
                    for key in class_keys:
                        if key in ['0', '1', 0, 1] or 'macro' in str(key) or 'weight' in str(key):
                            has_valid_keys = True
                            break
                    
                    if not has_valid_keys:
                        # Log the issue but don't crash
                        print(f"Warning: Classification report missing expected keys: {class_keys}")
                
    except Exception as e:
        flash(f"Error loading metrics: {str(e)}", "danger")
        print(f"Error loading metrics: {str(e)}")
    
    # Get list of available plots and copy them to static directory
    plots_path = os.path.join(eval_path, 'plots')
    available_plots = []
    static_plots_dir = os.path.join(STATIC_METRICS_DIR, 'plots')
    os.makedirs(static_plots_dir, exist_ok=True)
    
    if os.path.exists(plots_path):
        for plot_file in os.listdir(plots_path):
            if plot_file.endswith('.png'):
                try:
                    # Copy file to static directory if it doesn't exist or is outdated
                    src_file = os.path.join(plots_path, plot_file)
                    dst_file = os.path.join(static_plots_dir, plot_file)
                    
                    # Check if we need to copy the file (doesn't exist or source is newer)
                    copy_file = not os.path.exists(dst_file)
                    if not copy_file and os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        copy_file = True
                    
                    if copy_file:
                        shutil.copy2(src_file, dst_file)
                    
                    plot_name = plot_file.replace('.png', '').replace('_', ' ').title()
                    available_plots.append({
                        'name': plot_name,
                        'path': url_for('static', filename=f'metrics/plots/{plot_file}')
                    })
                except Exception as e:
                    print(f"Error processing plot {plot_file}: {str(e)}")
    
    # Group plots by category for better organization
    plot_categories = {
        'Basic Metrics': ['Confusion Matrix', 'Metrics Comparison'],
        'ROC & Precision-Recall': ['Roc Curve', 'Precision Recall Curve'],
        'Distribution Analysis': ['Probability Distribution', 'Error Distribution Pie'],
        'Advanced Analysis': ['Threshold Impact', 'Confusion Matrix Percentage', 'Performance Dashboard']
    }
    
    categorized_plots = {}
    for category, plot_names in plot_categories.items():
        categorized_plots[category] = []
        for plot in available_plots:
            if any(name in plot['name'] for name in plot_names):
                categorized_plots[category].append(plot)
    
    # Check if markdown report exists
    report_path = os.path.join(eval_path, 'evaluation_report.md')
    has_report = os.path.exists(report_path)
    report_content = ""
    
    if has_report:
        try:
            # Copy report to static directory
            static_report_path = os.path.join(STATIC_METRICS_DIR, 'evaluation_report.md')
            if not os.path.exists(static_report_path) or os.path.getmtime(report_path) > os.path.getmtime(static_report_path):
                shutil.copy2(report_path, static_report_path)
            
            with open(report_path, 'r') as f:
                report_content = f.read()
                
            # Convert relative URLs in report to absolute URLs
            try:
                report_content = report_content.replace(
                    'plots/', 
                    url_for('static', filename='metrics/plots/').replace('%2F', '/')
                )
            except Exception as e:
                print(f"Error converting URLs in report: {str(e)}")
        except Exception as e:
            report_content = f"Error loading report: {str(e)}"
            print(f"Error loading report: {str(e)}")
    
    return render_template('performance.html', 
                           metrics=metrics, 
                           available_plots=available_plots,
                           categorized_plots=categorized_plots,
                           has_report=has_report,
                           report_content=report_content)

# Load the model at startup
model = load_model()

if __name__ == '__main__':
    app.run(debug=True) 