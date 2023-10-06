# First, let's unzip the Archive.zip file to check its contents and identify the text files that need to be processed.
import zipfile
import os

# Define the unzip directory
unzip_dir = 'log_files'

# Create the directory if it doesn't exist
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

# List the files in the unzipped directory
unzipped_files = os.listdir(unzip_dir)

# Import required libraries for text manipulation and CSV writing
import re
import csv

# Function to clean and format a single text file
def process_text_file(file_path):
    # Initialize variables to hold data
    processed_data = []
    
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Initialize temporary variables to hold data for each question block
    temp_block = {}
    temp_answers = []
    collect_answers = False
    
    # Loop through each line in the text file
    for line in lines:
        if "<Question>" in line:
            # Save previous block if it exists
            if temp_block:
                processed_data.append(temp_block)
            
            # Reset temporary variables
            temp_block = {}
            temp_answers = []
            collect_answers = False
            
            # Store the question
            temp_block["Question"] = line.replace("<Question>", "").strip()
            
        elif "<Confidence>" in line:
            # Extract the max confidence value
            max_confidence = re.search(r"max:\s*([\d.]+)", line)
            if max_confidence:
                temp_block["Max Confidence"] = round(float(max_confidence.group(1)), 2)
        
        elif "<Answers>" in line:
            # Start collecting answers
            collect_answers = True
            # temp_answers.append(line.strip())
        
        elif "<Valid Indexes>" in line:
            # Stop collecting answers and store them
            collect_answers = False
            temp_answers = list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(temp_answers[:-1])))
            temp_answers.append('')
            temp_answers.append(line.strip())
            temp_block["Answers Enumeration"] = "\n".join(temp_answers)
        
        elif "<Final Answer>" in line:
            temp_block["Final Answer"] = line.replace("<Final Answer>", "").strip()
        
        elif "<Average DoX>" in line:
            avg_dox = re.search(r"<Average DoX>\s*([\d.]+)", line)
            if avg_dox:
                temp_block["Average DoX"] = round(float(avg_dox.group(1)), 2)
        
        elif "<Compliance score>" in line:
            compliance_score = re.search(r"<Compliance score>\s*([\d.]+)", line)
            if compliance_score:
                temp_block["Compliance Score"] = round(float(compliance_score.group(1)), 2)
        
        elif collect_answers:
            temp_answers.append(line.strip())
    
    # Save the last block if it exists
    if temp_block:
        processed_data.append(temp_block)
    
    return processed_data

# Initialize list to hold processed data for each file
all_processed_data = {}

# Process each text file in the unzipped directory
for file_name in unzipped_files:
    full_path = os.path.join(unzip_dir, file_name)
    all_processed_data[file_name] = process_text_file(full_path)

# Create CSV files based on the processed data
csv_file_paths = []
for file_name, data in all_processed_data.items():
    csv_file_path = f'{file_name.replace(".txt", ".csv")}'
    csv_file_paths.append(csv_file_path)
    
    # Define CSV columns
    csv_columns = ["Question", "Compliance Score", "Average DoX", "Max Confidence", "Final Answer", "Answers Enumeration"]
    
    # Write data to CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

