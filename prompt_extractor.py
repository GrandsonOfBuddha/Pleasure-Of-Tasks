#!/usr/bin/env python3
"""
Prompt Extractor for GPT Task Choice Experiment
Extracts all prompts from v4.py and writes them to a file
"""

from pathlib import Path
import sys

# Import from v4.py
sys.path.append(str(Path(__file__).parent))
from main import TASK_SPECS

def main():
    """Extract all prompts from TASK_SPECS and write to a file."""
    output_path = Path("all_prompts.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pair_number in range(1, len(TASK_SPECS) + 1):
            spec = TASK_SPECS[pair_number]
            
            # Extract the required information
            task1 = spec["task1"]
            task2 = spec["task2"]
            free_prompt = spec["free"]
            reverse_free_prompt = spec["reverse_free"]
            forced1_prompt = spec["forced1"]
            forced2_prompt = spec["forced2"]
            
            # Write in the specified format
            f.write(f"Task Pair {pair_number}\n")
            f.write(f"Task 1: {task1}\n")
            f.write(f"Task 2: {task2}\n")
            f.write("Free-choice Prompt:\n")
            f.write(f"{free_prompt}\n")
            f.write("Reverse Free-choice Prompt:\n")
            f.write(f"{reverse_free_prompt}\n")
            f.write("Forced-task 1 Prompt:\n")
            f.write(f"{forced1_prompt}\n")
            f.write("Forced-task 2 Prompt:\n")
            f.write(f"{forced2_prompt}\n")
            
            # Add a blank line between pairs (except after the last pair)
            if pair_number < len(TASK_SPECS):
                f.write("\n")
    
    print(f"Prompt export complete: all_prompts.txt")

if __name__ == "__main__":
    main()
