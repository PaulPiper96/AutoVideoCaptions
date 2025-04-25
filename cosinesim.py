import torch
import os
import glob
import csv
import numpy as np
import imageio
import open_clip
from PIL import Image
import ast
import re

def distance(prompt, frame, model, preprocess, device):
    with torch.no_grad():
        image_pil = Image.fromarray(frame)
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        if model.visual.conv1.weight.dtype == torch.float16:
            image_tensor = image_tensor.half()

        text_tokens = open_clip.get_tokenizer('ViT-H-14')([prompt]).to(device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        print("Similarity:", similarity)
        return similarity

def readcsv(filename, arraytopass):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')

    for index, line in enumerate(lines):
        try:
            datapoint=[]
            parts = line.split(",", 1)
            caption_dict = ast.literal_eval(parts[1])
            caption_dict=caption_dict.replace("{'<CAPTION>':","")
            caption_dict=caption_dict.replace("'}","")
            caption_dict=caption_dict.replace("'","")
            print(caption_dict)
            datapoint.append(parts[0])
            datapoint.append(caption_dict)
            arraytopass.append(datapoint)
        except Exception as e:
            caption = f"[Error parsing line {index}: {e}]"

      

    return arraytopass

def writecsv(path, data):
    with open(path, "w", newline="") as csvfile:
        out = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for entry in data:
            if not isinstance(entry, (list, tuple)):
                entry = [entry]
            out.writerow(entry)


def iterate_images(directory):
    directory = directory.rstrip("/\\")
    image_extensions = ('*.mp4', '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.webp')
    locations = []

    for ext in image_extensions:
        pattern = os.path.join(directory, '**', ext)
        matches = glob.glob(pattern, recursive=True)

        # Filter files to only include those that match the _<number> pattern
        for file_path in matches:
            filename = os.path.basename(file_path)
            if re.search(r'_\d+\.', filename):  # matches things like _123.png or _10492.jpg
                locations.append(file_path)

    # Extract number after underscore and sort based on it
    def extract_number(filename):
        match = re.search(r'_(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else float('inf')

    locations.sort(key=extract_number)
    return locations

if __name__ == "__main__":
    prompts = []
    readcsv(r"captions.csv", prompts)

    bench_directory = r"frames"
    allimages = iterate_images(bench_directory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        device=device,
        precision='fp16'
    )
  
    cosinedata = []
    for i, entry in enumerate(prompts):
        if i >= len(allimages):  # prevent index error
            break

        image = Image.open(allimages[i]).convert("RGB")
        print(entry)
        similarity = distance(entry[1], np.array(image), model, preprocess, device)
        datap=[]
        datap.append(i)
        datap.append(entry[0])
        datap.append( entry[1])
        datap.append(similarity)
        cosinedata.append(datap)
        
    texttosafe=""    
    for anything in cosinedata:
      
        textinput=anything[1]
        textinput=textinput.replace('"', "")
        textinput=textinput.replace('frames\_', "Bild:")   
        textinput=textinput+"\n"
        textinput=textinput+anything[2] +"\n"      
        texttosafe+= textinput
    print(texttosafe)
    
    
    writecsv(r"checksimilarity.csv", cosinedata )

    with open("captions.txt", "a", encoding="utf-8") as file:
        file.write(texttosafe + "\n") 