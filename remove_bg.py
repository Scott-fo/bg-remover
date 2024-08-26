import os
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm


def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 4:  # If the image has an alpha channel, remove it
        im = im[:, :, :3]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(
        im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(
        result, size=im_size, mode='bilinear'), 0)
    ma, mi = torch.max(result), torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2,
                                      0).cpu().data.numpy().astype(np.uint8)
    return np.squeeze(im_array)


def remove_background(model, device, input_path: str, output_path: str):
    orig_im = np.array(Image.open(input_path).convert('RGB'))
    orig_im_size = orig_im.shape[0:2]
    model_input_size = [1024, 1024]
    image = preprocess_image(orig_im, model_input_size).to(device)
    with torch.no_grad():
        result = model(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.fromarray(orig_im)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save(output_path, format='PNG')


def process_image(args):
    model, device, input_path, output_path = args
    try:
        remove_background(model, device, input_path, output_path)
        return f"Processed: {input_path} -> {output_path}"
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"


def main(input_dir: str, output_dir: str, num_workers: int = 5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)

    print(f"Found {total_images} images to process.")
    print(f"Using device: {device}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(
                output_dir, f"processed_{os.path.splitext(img_file)[0]}.png")
            futures.append(executor.submit(
                process_image, (model, device, input_path, output_path)))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            print(future.result())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
