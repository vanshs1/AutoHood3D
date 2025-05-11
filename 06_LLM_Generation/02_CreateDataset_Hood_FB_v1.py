#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction".

This is a demonstration intended to provide a working example with data provided in the repo.
For application on other datasets, the requirement is to configure the settings. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python <this_file.py>" will run the demo code. 
    NOTE - It is important to check README prior to running this code.

Contact: 
    - Vansh Sharma at vanshs@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""

import os, json, re
import random
from transformers import pipeline
import multiprocessing as mp
random.seed(4388)
from datasets import Dataset, Features, Value, Image as DatasetsImage
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

#%% Functions
def get_description(records, target_name):
    """
    Scan the list for the entry with hood_name == target_name
    and return its description (or None if not found).
    """
    for entry in records:
        if entry['hood_name'] == target_name:
            return entry['description']
    return None

def extract_hood_num(hood_name):
    """
    Grabs digits after 'sk' in hood_name.
    E.g. 'convHull_sk005_geo4' -> '005'
    """
    m = re.search(r'sk(\d+)', hood_name)
    return m.group(1) if m else None

def extract_geo_num(file_name):
    """
    Grabs digits after 'geo_' in file_name.
    E.g. 'geo_005_clusterID_...' -> '005'
    """
    m = re.search(r'geo_(\d+)', file_name)
    return m.group(1) if m else None

def hood_matches_file(hood_name, file_name):
    """
    Compare hood and geo numbers as zero‐padded 3‐digit strings.
    """
    hn = extract_hood_num(hood_name)
    gn = extract_geo_num(file_name)
    if hn is None or gn is None:
        return False
    return hn.zfill(3) == gn.zfill(3)

def parse_Hoodparams(name):
    """
    From a string like:
      geo_005_clusterID_0_crvCount_3_0001_0002_0003_cd_0.034_md_0.018.stl
    extract and return a variation description string.
    """
    base = name.rsplit('.', 1)[0]
    m = re.search(
        r'crvCount_(?P<crvCount>\d+).*_cd_(?P<cd>\d+\.\d+)_md_(?P<md>\d+\.\d+)',
        base
    )
    if not m:
        raise ValueError(f"Unexpected format: {name!r}")
    gd = m.groupdict()
    crv = int(gd['crvCount'])
    md = float(gd['md'])
    cd = float(gd['cd'])
    # Variation details text
    return (
        f"Variations include: curve count of {crv} "
        f"(total of {2*crv} due to symmetry), "
        f"minimum curve distance of {md} and center distance of {cd}"
    )

def extract_user_input(text: str) -> str:
    """
    Extract the text under "## 1. **User Input**" up to "## 2.".
    Falls back gracefully if one of the markers is missing.
    """
    start_marker = "## 1. **User Input**"
    end_marker   = "## 2."

    start = text.find(start_marker)
    if start == -1:
        return ""  # no User Input section found
    # move past the marker
    start += len(start_marker)

    end = text.find(end_marker, start)
    if end == -1:
        # no second section, grab till end of text
        return text[start:].strip()
    return text[start:end].strip()

def build_meta_prompt(description_template: str, variation_details: str) -> str:
    """
    Construct the meta‐prompt for the Vision‐LLM training.
    """
    return f"""
    You are a CAD‐focused Vision‐LLM. Analyze the image provided along with these guidelines:
    
    Description of the base geometry without cuts:
    \"{description_template}\"
    
    Variation Details:
    {variation_details}
    
    Produce exactly three labeled sections—no additional commentary:
    
    1. **User Input**  
       Formulate the user request using the base description and variation details.
    
    2. **Chain of Thought**  
       Show internal reasoning step by step:
       - Start with: “Let me think through the requirements…”  
       - Parse counts, distances, symmetry from the Variation Details.  
       - Identify inner vs. outer face features from the base description.  
       - Plan point‐cloud density and cut placement.
    
    3. **Solution**  
       Provide a placeholder for the point‐cloud output:  
       “Solution: {{point_cloud}}
       
       """

#%% -- Worker and Parallel Execution --
def worker(task_list, gpu_id, output_dir):
    device = f"cuda:{gpu_id}"
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device=device,
        torch_dtype="bfloat16"
    )
    # Disable the fused SDPA path so we fall back to the safe PyTorch implementation
    pipe.model.config.use_sdp_attention = False
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"final_prompts_gpu{gpu_id}.json")

    # Build HF Dataset
    examples = [{"prompt": p, "image": {"path": i}} for p, i in task_list]
    ds = Dataset.from_list(
        examples,
        features=Features({
            "prompt": Value("string"),
            "image": DatasetsImage()
        })
    )

    def infer_batch(batch):
        """
        Batch inference function for HuggingFace Dataset.map,
        accepting PIL images directly from the 'image' column.
        """
        # Extract prompt strings
        prompts = batch["prompt"]  # list of strings
        pil_images = batch["image"]  # list of PIL.ImageFile instances
        
        batched_messages = [
            [
                {"role": "system", "content": [{"type": "text",  "text": prompt}]},
                {"role": "user",   "content": [{"type": "image", "path": image }]}
            ]
            for prompt, image in zip(prompts, pil_images)
        ]

    
        # Run the pipeline once on the entire batch
        outputs = pipe(
            text=batched_messages,
            max_new_tokens=2000
        )
        
        user_inputs  = []
        full_outputs = []
        # Handle possible nested list outputs
        for resp in outputs:
            # If the pipeline returns a list for each conversation, take the first element
            if isinstance(resp, (list, tuple)):
                resp = resp[0]
            # Extract the final generated text
            gen_list = resp.get("generated_text", [])
            if gen_list:
                txt = gen_list[-1].get("content", "")
            else:
                txt = ""
    
            user_inputs.append(extract_user_input(txt))
            full_outputs.append(txt)
    
        return {
            "geometry": pil_images,
            "user_input":  user_inputs,
            "full_output": full_outputs
        }

    result = ds.map(
        infer_batch,
        batched=True,
        batch_size=1,  # increase if your GPU can handle more
        remove_columns=["prompt","image"],
        desc=f"GPU {gpu_id} Inference"
    )

    # # Write the JSON
    # with open(out_file, "w") as fw:
    #     json.dump(result, fw, indent=2)
    
    # OPTION A: Write with HF’s to_json
    # result.to_json(out_file, orient="records", lines=True)

    # OPTION B: Convert and dump manually
    data_dict = result.to_dict()
    records = [
    { key: data_dict[key][i] for key in data_dict }
    for i in range(len(next(iter(data_dict.values()))))
    ]
    with open(out_file, "w", encoding="utf-8") as fw:
        json.dump(records, fw, indent=2, ensure_ascii=False)

def parallel_process(tasks, output_dir, num_gpus=7, start_gpu=1):
    """
    Distribute tasks across GPUs start_gpu ... start_gpu+num_gpus-1.
    """
    gpu_ids = list(range(start_gpu, start_gpu + num_gpus))
    task_lists = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, task in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        task_lists[gpu_id].append(task)

    processes = []
    for gpu_id, sub_tasks in task_lists.items():
        p = mp.Process(target=worker, args=(sub_tasks, gpu_id, output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

#%%
if __name__ == "__main__":
    # Paths (customize as needed)
    stlImgDir = "./sample_data"
    output_dir = "Output_folder"
    baseJSON = "baseSkins_prompts.json"

    # Load base descriptions
    with open(baseJSON, 'r') as f:
        records = json.load(f)
    num_to_desc = { extract_hood_num(rec['hood_name']): rec['description']
                    for rec in records }

    # Gather image tasks
    hood_dirs = sorted([
        d for d in os.listdir(stlImgDir)
        if d.startswith("geo_")
    ])
    hood_dirs = hood_dirs[0:2]
    tasks = []
    for hood_dir in hood_dirs:
        img_folder = os.path.join(stlImgDir, hood_dir)
        images = sorted([
            fn for fn in os.listdir(img_folder)
            if fn.lower().endswith(".png")
        ])
        images = images[0:8]
        geo_num = extract_geo_num(hood_dir)
        if geo_num not in num_to_desc:
            continue
        base_desc = num_to_desc[geo_num]
        for img in images:
            variation = parse_Hoodparams(img)
            prompt = build_meta_prompt(base_desc, variation)
            img_path = os.path.join(img_folder, img)
            tasks.append((prompt, img_path))
    
    # Run in parallel on GPUs
    parallel_process(tasks, output_dir, num_gpus=8, start_gpu=0)






    
    
