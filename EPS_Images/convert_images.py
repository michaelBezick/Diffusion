from PIL import Image
import os

def convert_image_to_eps(image_path, output_path):
    with Image.open(image_path) as img:
        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
        img.save(output_path, format='EPS')

def find_images(directory, extensions=(".png", ".jpg", ".jpeg")):
    matches = []
    for root, dirs, files in os.walk(directory):
        if "/." in root:
            continue
        for file in files:
            if file.lower().endswith(extensions):
                matches.append(os.path.join(root, file))

    return matches

search_directory = "../"

for image in find_images(search_directory):
    if "Experiment_3" in image or "Experiment_6" in image:
        print("Continuing")
        continue
    if "Scatter" in image:
        continue
    print("------------------------------")
    print(image)
    save_dir = image.split("/")[-1].split(".")[0] + ".eps"
    print(save_dir)
    convert_image_to_eps(image, save_dir)
