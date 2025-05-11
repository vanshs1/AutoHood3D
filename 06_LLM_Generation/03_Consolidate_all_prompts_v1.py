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
import os
import json
import re
import pandas as pd
#%%

def extract_sections(text):
    """
    Given a single full_output string, return a dict with the
    user_input, chain_of_thought, and solution sections.
    """
    sections = {"user_input": "", "chain_of_thought": "", "solution": ""}
    
    # 1. User Input: everything between "1. **User Input**" and "2."
    m = re.search(r"1\.\s*\*\*User Input\*\*(.*?)(?=2\.)", text, re.DOTALL)
    if m:
        sections["user_input"] = m.group(1).strip()
    
    # 2. Chain of Thought: between "2." and "3."
    m = re.search(r"2\.\s*\*\*Chain of Thought\*\*(.*?)(?=3\.)", text, re.DOTALL)
    if m:
        sections["chain_of_thought"] = m.group(1).strip()
    
    # 3. Solution: after "3. **Solution**"
    m = re.search(r"3\.\s*\*\*Solution\*\*(.*)", text, re.DOTALL)
    if m:
        sections["solution"] = m.group(1).strip()
    
    return pd.Series(sections)

#%%
# Directory with outputs created by the LLM
root_dir = "output_final_prompts"

records = []
for dirpath, _, filenames in os.walk(root_dir):
    for fn in filenames:
        if fn.lower().endswith(".json"):
            full_path = os.path.join(dirpath, fn)
            with open(full_path, 'r') as f:
                data = json.load(f)
            # Handle JSON being a list or a single dict
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                geom = entry.get("geometry", {})
                path = geom.get("path") if isinstance(geom, dict) else None
                if path is not None:
                    path = path.split('/')[-1][:-4]
                full_output = entry.get("full_output", "")
                records.append({
                    "geometry_id": path,
                    "full_output": full_output
                })
df = pd.DataFrame(records)
new_cols = df["full_output"].apply(extract_sections)
df = pd.concat([df, new_cols], axis=1)

df.drop('full_output', axis=1, inplace=True)

df = df.sort_values(by="geometry_id").reset_index(drop=True)

df.to_json("consolidated_prompts.jsonl", 
           orient="records", indent=2,
           lines=True)



