from PIL import Image

# Load the uploaded image
input_image_path = "/u1/a2soni/DynamiCrafter/prompts/1024/robot01.png"
output_image_path = "/u1/a2soni/DynamiCrafter/prompts/512/robot02.png"

# Open the image using PIL
original_image = Image.open(input_image_path)

# Resize the image to 320x512
resized_image = original_image.resize((512, 320))

# Save the resized image
resized_image.save(output_image_path)
