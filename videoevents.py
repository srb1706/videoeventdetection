import cv2
import openai
import os
from typing import List
from PIL import Image
from dotenv import load_dotenv
import io, base64
import csv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# events of interest
EVENT_TYPES = [
    "Manual bin lifting",
    "Vehicle blocking path",
    "Bin overflowing or missing",
    "Near collision",
    "Truck arm malfunction",
    "Worker unsafe behavior",
]

# Step 1: Extract frames every N seconds
def extract_frames(video_path: str, interval_sec: int = 3) -> List[tuple]:
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    interval_frames = interval_sec * fps
    success, frame = vidcap.read()
    count = 0
    frames = []

    while success:
        if count % interval_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            timestamp = count / fps  # timestamp in seconds
            frames.append((pil_image, timestamp))
        success, frame = vidcap.read()
        count += 1

    vidcap.release()
    return frames

# Step 2: Use GPT-4.1 mini to caption each frame
def caption_image(img: Image.Image) -> str:
    
    # Convert the PIL image to binary data
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)
    
      
    # Encode the image as base64
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    
    # OpenAI API call to caption the image
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes street-level dashcam footage from a garbage truck."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this scene in detail, focusing on any events relevant to waste collection or safety."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ],
            },
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

# Step 3: Use GPT-4.1 mini to classify waste-related events from caption
def classify_events(caption: str) -> List[str]:
    prompt = f"""
You are a waste management safety assistant with an ability to identify interesting events to flag to safety and operations managers for review. Analyze the following video description and return a list of relevant events from this list:

{", ".join(EVENT_TYPES)}

Video Caption:
\"{caption}\"

Return only the relevant events that are clearly observed.
"""
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip().split("\n")

# Step 4: Run pipeline on a video
def process_video(video_path: str):
    frames = extract_frames(video_path)
    
    # Prepare CSV file
    csv_filename = f"event_detection_results.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame_Number', 'Timestamp_Seconds', 'Timestamp_MMSS', 'Detected_Event'])
        
        for i, (frame, timestamp) in enumerate(frames):
            print(f"\n--- Frame {i} ---")
            caption = caption_image(frame)
            print(f"Caption: {caption}")
            events = classify_events(caption)
            
            # Convert timestamp to MM:SS format
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_mmss = f"{minutes:02d}:{seconds:02d}"
            
            print(f"Timestamp: {timestamp_mmss}")
            print("Detected Events:")
            
            if events and events[0].strip():  # Check if there are actual events
                for event in events:
                    event = event.strip()
                    if event and event != "None" and not event.startswith("No"):
                        print(f"• {event}")
                        writer.writerow([i, f"{timestamp:.2f}", timestamp_mmss, event])
            else:
                print("• No events detected")
                writer.writerow([i, f"{timestamp:.2f}", timestamp_mmss, "No events detected"])
    
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    process_video("videoplayback.mp4")
