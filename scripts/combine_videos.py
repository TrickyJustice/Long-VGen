from moviepy.editor import VideoFileClip, concatenate_videoclips

def combine_videos(video1_path, video2_path, output_path):
    """
    Combines two MP4 videos into one.

    Args:
        video1_path (str): Path to the first video.
        video2_path (str): Path to the second video.
        output_path (str): Path to save the combined video.
    """
    # Load the video clips
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)
    
    # Combine the clips
    combined = concatenate_videoclips([clip1, clip2])
    
    # Write the combined video to the output file
    combined.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    print(f"Combined video saved to {output_path}")

# Usage example
video1_path = "/u1/a2soni/DynamiCrafter/scripts/combined_video_32_frames.mp4"
video2_path = "/u1/a2soni/DynamiCrafter/results/dynamicrafter_512_seed123/samples_separate/frame_28_sample0.mp4"
output_path = "combined_video_48_frames.mp4"

combine_videos(video1_path, video2_path, output_path)
