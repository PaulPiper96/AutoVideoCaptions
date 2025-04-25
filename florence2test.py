from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import torch
import csv
import os
import re
import glob

    
def writecsv(path, data):
    # Open CSV file in text mode with newline=''
    with open(path, "w", newline="") as csvfile:
        out = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for entry in data:
            # Check if the entry is not an iterable (or is a string, which is iterable but we want to treat as a single value)
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

def run_example(task_prompt, image_path_or_url, text_input=None):
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Check if the input is a URL or local path
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")  # local file

    prompt = task_prompt if text_input is None else task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


if __name__ =="__main__":
    #
    task_prompt = '<DETAILED_CAPTION>'
    promptarray=[]
    
    imagedirs=r"frames"
    imageurls= iterate_images(imagedirs)
    ''''''
    for dir in imageurls:
        print(dir)
        datapoint=[]
        datapoint.append(dir)
        datapoint.append(run_example(task_prompt, dir))
        promptarray.append(datapoint)
    writecsv(r"captions.csv", promptarray)