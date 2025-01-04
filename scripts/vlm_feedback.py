from openai import OpenAI
from PIL import Image
import os
import base64
import json
import traceback
import cv2


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

import os
import base64

def encode_frames_from_directory(directory_path):
    # List all files in the directory
    frame_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    print(frame_files)
    
    # Sort the frames by the integer value in the filename
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(frame_files)
    encoded = []
    cache_dir = "feedback_fewshot/.cache/"
    
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    for frame_file in frame_files:
        frame_path = os.path.join(directory_path, frame_file)
        # Copy file to cache (optional, depending on usage)
        cached_file = os.path.join(cache_dir, frame_file)
        os.system(f'cp "{frame_path}" "{cached_file}"')
        
        # Encode the frame in base64
        with open(frame_path, "rb") as image:
            encoded.append(base64.b64encode(image.read()).decode('utf-8'))
    
    return encoded


def chat_with_openai(gif_paths, text_prompts):
    api_key = ''
    # base64_image8 = encode_image("feedback_fewshot/assembly_24_test.png")
    # base64_image8_out = encode_gif("feedback_fewshot/assembly_checkpoint_276_out_3.gif")
    # #print(len(base64_image8_out))
    # base64_image7 = encode_image("feedback_fewshot/assembly.png")
    # base64_image7_out = encode_gif("feedback_fewshot/assembly_out_1iter_1.gif")
    # base64_image6_out = encode_gif("feedback_fewshot/basketball_out_1iter_00.gif")
    # base64_image6 = base64_image6_out[0]
    # base64_image5_out = encode_gif("feedback_fewshot/hammer_checkpoint_310_out_0iter_7.gif")
    # base64_image5 = base64_image5_out[0]
    # Initialize a conversation with the system role description
    
    messages_c=[
        {
            "role": "user",
            "content": [{
                            "type": "text",
                            "text": """ 
                            You are an AI assistant specialized in improving temporal coherence in video sequences. Analyze the following two images for temporal inconsistencies related to motion or object continuity:

                            Image A: [Last frame of the current sequence]
                            Image B: [First frame of the next sequence]
                            Instructions:

                            Focus only on temporal inconsistencies such as sudden changes in object positions, motion discontinuities, or abrupt transitions.
                            Do not consider lighting, color variations, or background changes unless they directly impact temporal consistency.
                            Provide one concise suggestion (maximum 30 words) to improve temporal coherence between the sequences.
                            Do not mention specific frames or images in your suggestion.
                            Frame your suggestion generally so it can be applied by a video generation model without direct access to the images.
                            Example of Desired Output:

                            "Adjust object trajectories to ensure smooth motion continuity between sequences."
                            Now, please provide your one-sentence suggestion to enhance temporal coherence.
                                    """
                            }
                        
            ]
        }
    ]
    responses = []

    # Loop through each example provided
    for gif_path, text_prompt in zip(gif_paths, text_prompts):
        # Conducting the conversation for each set of inputs
        while True:
            try:
                client = OpenAI(api_key=api_key)
                # base_image = encode_image(image_path)
                base64_image_out = encode_frames_from_directory(gif_path)
                messages_q=[
                    {
                        "role": "user", 
                        "content": [
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[0]}"
                                            },
                                        },
                                    {
                                        "type":"image_url", 
                                        "image_url":  {
                                            "url": f"data:image/jpeg;base64,{base64_image_out[1]}"
                                            },
                                        }
                            ]
                        },
                        {"role": "user", "content":"the conditioning image is the first upload, the next seven uploads are the key frames of the gif and the textual prompt is:" + text_prompt + ".Return only final feedback"},
                ]
                response = client.chat.completions.create(model="gpt-4o",messages=messages_c + messages_q)


                # Store and print feedback
                feedback = response.choices[0].message.content
                responses.append(feedback)
                print("Feedback received:", feedback)
                break
            except:
                print("Bad request, retrying")
                traceback.print_exc()
                continue

    return responses

if __name__ == "__main__":
    # Example API Key and paths
      
    # image_paths = ['feedback_fewshot/assembly_24_test.png']
    gif_paths = ['/u1/a2soni/DynamiCrafter/Saved-images/']
    text_prompts = ['a robot is walking through a destroyed city']
    
    feedback_responses = chat_with_openai(gif_paths, text_prompts)

    with open("robot_video_ratings.json", "a") as final:
        final.write("1:")
        json.dump(feedback_responses, final)
        final.write("\n")  # Add a newline after each JSON dump

    print("All responses:", feedback_responses)
