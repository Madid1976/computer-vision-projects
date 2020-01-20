import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Loads an image from the specified path.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
    return img

def save_image(image, output_path):
    """
    Saves an image to the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def resize_image(image, width, height):
    """
    Resizes an image to the specified width and height.
    """
    if image is None:
        return None
    resized_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img

def convert_to_grayscale(image):
    """
    Converts a color image to grayscale.
    """
    if image is None:
        return None
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """
    Applies Gaussian blur to an image.
    """
    if image is None:
        return None
    blurred_img = cv2.GaussianBlur(image, kernel_size, sigmaX)
    return blurred_img

if __name__ == "__main__":
    # Create a dummy image for testing
    dummy_image_path = "/home/ubuntu/computer-vision-projects/data/dummy_image.png"
    dummy_output_path = "/home/ubuntu/computer-vision-projects/data/processed_image.png"
    
    # Generate a simple red square image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[:, :, 2] = 255 # Red channel
    save_image(dummy_image, dummy_image_path)

    # Example Usage
    img = load_image(dummy_image_path)

    if img is not None:
        print(f"Original image shape: {img.shape}")

        # Resize image
        resized_img = resize_image(img, 50, 50)
        print(f"Resized image shape: {resized_img.shape}")
        save_image(resized_img, dummy_output_path.replace(".png", "_resized.png"))

        # Convert to grayscale
        gray_img = convert_to_grayscale(img)
        print(f"Grayscale image shape: {gray_img.shape}")
        save_image(gray_img, dummy_output_path.replace(".png", "_gray.png"))

        # Apply Gaussian blur
        blurred_img = apply_gaussian_blur(img)
        print(f"Blurred image shape: {blurred_img.shape}")
        save_image(blurred_img, dummy_output_path.replace(".png", "_blurred.png"))

        print("Image processing examples completed!")
