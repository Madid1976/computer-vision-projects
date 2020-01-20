import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

def load_image_for_classification(image_path):
    """
    Loads and transforms an image for classification using a pre-trained model.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error loading or transforming image {image_path}: {e}")
        return None

def classify_image(image_tensor, model, class_names):
    """
    Classifies an image using a pre-trained model.
    """
    if image_tensor is None:
        return "", 0.0

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 1)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item()
    return predicted_class, confidence

def get_imagenet_class_names():
    """
    Fetches ImageNet class names (simplified for example).
    In a real scenario, you'd load a proper class_idx_to_name mapping.
    """
    # This is a highly simplified list for demonstration. 
    # A real implementation would load a comprehensive list from a file or URL.
    return [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
        "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch",
        "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie",
        "chickadee", "kinglet", "water ouzel", "shrike", "cedar waxwing", "red-winged blackbird",
        "grackle", "hornbill", "cardinal", "eagle", "vulture", "owl", "falcon", "parrot",
        "dog", "wolf", "cat", "lion", "tiger", "bear", "zebra", "pig", "cow", "horse",
        "elephant", "monkey", "kangaroo", "squirrel", "hedgehog", "otter", "snake", "lizard",
        "turtle", "spider", "snail", "crab", "lobster", "bird", "fish", "insect", "flower",
        "tree", "car", "truck", "boat", "airplane", "chair", "table", "lamp", "couch",
        "bed", "toilet", "computer", "keyboard", "mouse", "book", "cup", "bottle", "fork",
        "knife", "spoon", "banana", "apple", "orange", "sandwich", "pizza", "burger",
        "hotdog", "donut", "cake", "coffee", "tea", "milk", "water", "beer", "wine",
        "backpack", "umbrella", "handbag", "tie", "watch", "clock", "scissors", "teddy bear",
        "hair dryer", "toothbrush"
    ]

if __name__ == "__main__":
    # Create a dummy image for testing
    dummy_image_path = "/home/ubuntu/computer-vision-projects/data/dummy_cat.jpg"
    try:
        from PIL import ImageDraw
        img = Image.new("RGB", (224, 224), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,10), "Dummy Cat", fill=(255,255,0))
        os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
        img.save(dummy_image_path)
        print(f"Dummy image created at {dummy_image_path}")
    except ImportError:
        print("Pillow not installed. Cannot create dummy image. Please install with `pip install Pillow`")
        print("Skipping image classification example.")
        exit()

    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    class_names = get_imagenet_class_names()

    # Load and classify the dummy image
    image_tensor = load_image_for_classification(dummy_image_path)
    if image_tensor is not None:
        predicted_class, confidence = classify_image(image_tensor, model, class_names)
        print(f"\nPredicted class: {predicted_class} with confidence: {confidence:.4f}")
        assert predicted_class == "cat" or predicted_class == "tiger" or predicted_class == "lion", "Classification might be off for dummy image."
        print("Image classification example completed successfully!")
