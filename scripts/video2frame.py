import cv2

def extract_frames(video1_path, video2_path):
    """
    Extracts the last 8 frames from the first video and the first 8 frames from the second video.
    
    Args:
        video1_path (str): Path to the first MP4 video.
        video2_path (str): Path to the second MP4 video.
        
    Returns:
        list: A list of 16 frames (last 8 from video1 and first 8 from video2).
    """
    frames = []
    
    # Process the first video
    cap1 = cv2.VideoCapture(video1_path)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Move to the frame index where the last 8 frames start
    start_frame1 = max(0, frame_count1 - 16)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame1)
    
    for _ in range(16):
        ret, frame = cap1.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap1.release()
    
    # Process the second video
    cap2 = cv2.VideoCapture(video2_path)
    for _ in range(16):
        ret, frame = cap2.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap2.release()
    
    return frames


video1_path = "/u1/a2soni/DynamiCrafter/results/dynamicrafter_512_seed123/samples_separate/robot_video_9_sample0.mp4"
video2_path = "/u1/a2soni/DynamiCrafter/results/dynamicrafter_512_seed123/samples_separate/robot_video_12_sample0.mp4"

frames = extract_frames(video1_path, video2_path)

# Example: Save the frames as images (optional)
for i, frame in enumerate(frames):
    cv2.imwrite(f"/u1/a2soni/DynamiCrafter/Saved-images/frame_{i + 1}.jpg", frame)