import json
import os
import glob
from pathlib import Path


def combine_json_files(input_pattern, output_file):
    """
    Combines multiple COCO-style JSON annotation files into a single file.
    
    Args:
        input_pattern: Glob pattern to match the input JSON files (e.g., "annotations/*.json")
        output_file: Path to the output combined JSON file
    """
    # Initialize combined data structure
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track image and annotation IDs to avoid duplicates
    image_ids_seen = set()
    annotation_ids_seen = set()
    next_image_id = 1
    next_annotation_id = 1
    skip_file_list = []
    
    # Find all matching JSON files
    json_files = glob.glob(input_pattern)
    print(f"Found {len(json_files)} JSON files to combine")

    
    # Process each file
    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Store categories from the first file
        if i == 0 and data.get("categories"):
            combined_data["categories"] = data["categories"]
        
        # empty annos
        if len(data['annotations']) == 0:
            skip_file_list.append(data['images'][0]['id'])
            continue
        
        # Process images
        for image in data.get("images", []):
            # # Check if image ID already exists
            # if image["id"] in image_ids_seen:
                # old_id = image["id"]
                # image["id"] = next_image_id
                # next_image_id += 1
                
                # # Update annotations that reference this image
                # for ann in data.get("annotations", []):
                #     if ann["image_id"] == old_id:
                #         ann["image_id"] = image["id"]
            
            image_ids_seen.add(image["id"])
            image['file_name'] = image['file_name'].replace('.png', '.jpg')
            combined_data["images"].append(image)
        
        # Process annotations
        for annotation in data.get("annotations", []):
            # Check if annotation ID already exists
            if annotation["id"] in annotation_ids_seen:
                annotation["id"] = next_annotation_id
                next_annotation_id += 1
            
            annotation_ids_seen.add(annotation["id"])
            combined_data["annotations"].append(annotation)
    
    # Write combined data to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    if len(skip_file_list):
        with open(output_path.parent / f"{output_path.name.split('.')[0]}_skipped.txt", 'w') as f:
            f.write("\n".join([str(x) for x in skip_file_list]))
    
    print(f"Combined data written to {output_file}")
    print(f"Total images: {len(combined_data['images'])}")
    print(f"Total annotations: {len(combined_data['annotations'])}")
    print(f"Total categories: {len(combined_data['categories'])}")
    print(f"Skipped images: {len(skip_file_list)}")
    
    return combined_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple COCO-style JSON annotation files")
    parser.add_argument("--input", required=True, help="Glob pattern for input JSON files, e.g., 'annotations/*.json'")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    combine_json_files(f"{args.input}/*.json", args.output)