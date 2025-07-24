import re 
import os
import json

project_root = os.path.dirname(__file__)  
data_dir = os.path.join(project_root, "2025-07-23+-+Data+for+Julianna")
output_dir = os.path.join(project_root, "combined_views_tracks")
os.makedirs(output_dir, exist_ok=True)


pattern = re.compile(r"^([0-9a-fA-F]+) - View (\d+)$")
video_views = {}

# Build dictionary of video_id -> set of views
for folder in os.listdir(data_dir):
    match = pattern.match(folder) 
    if match: 
        video_id, view = match.groups()
        if video_id not in video_views: 
            video_views[video_id] = set()
        video_views[video_id].add(view)

# Find video_ids with only one view
single_view = []
for video_id,views in video_views.items(): 
    if len(views) == 1: 
        single_view.append(video_id)

single_view = set(single_view) #speed up 
     
# Loop through all video_ids
for video_id in video_views: 
    if video_id not in single_view: 
        # Handle videos with both views
        combined_json_data = {
            "Framerate (FPS)" : None,
            "Tracks":[],
            "um per Pixel": None
        }
        for view in ["1", "2"]:
            folder_name = f"{video_id} - View {view}"
            folder_path = os.path.join(data_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if filename.endswith("tracks_with_mot_params.json"):
                    file_path = os.path.join(folder_path, filename)
                    try: 
                        with open(file_path, "r") as f: 
                            data = json.load(f)
                            
                            #append all tracks 
                            if "Tracks" in data: 
                                combined_json_data["Tracks"].extend(data["Tracks"])
                            #set framerate if not specified
                            if combined_json_data["Framerate (FPS)"] is None:
                                combined_json_data["Framerate (FPS)"] = data.get("Framerate (FPS)") #retrieves the float
                                
                            if combined_json_data["um per Pixel"] is None: 
                                combined_json_data["um per Pixel"] = data.get("um per Pixel")
                    except Exception as e: 
                        print(f"Error reading {file_path}: {e}")
                        
        output_path = os.path.join(output_dir, f"{video_id}_combined.json")
        with open(output_path, "w") as out_f:
            json.dump(combined_json_data, out_f, indent=2)
        print(f"Saved combined file for {video_id} to {output_path}")

    else: 
        # Handle single-view videos
        view = next(iter(video_views[video_id]))
        folder_name = f"{video_id} - View {view}"
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith("tracks_with_mot_params.json"):
                file_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_dir, f"{video_id}.json")
                try: 
                    with open(file_path, "r") as f: 
                        data = json.load(f)
                    with open(output_path, "w") as out_f: 
                        json.dump(data, out_f, indent=2) 
                    print(f"Saved single-view for {video_id} to {output_path}")
                except Exception as e: 
                    print(f"Error processing {file_path}: {e}")


# ensure matching 
all_folder_matches = [
    folder for folder in os.listdir(data_dir)
    if pattern.match(folder)
]
total_matched_folders = len(all_folder_matches)
#count number of unique videoIds 
total_video_ids = len(video_views)

total_single_view = len(single_view)
total_two_view = total_video_ids - total_single_view

output_files = [
    f for f in os.listdir(output_dir)
    if f.endswith(".json")
    ]
total_output_files = len(output_files)

# print summar
print(f"Total folders in original data package: {total_matched_folders}")
print(f"Unique video ids: {total_video_ids}")
print(f"Videos with single view: {total_single_view}")
print(f"Videos with two views: {total_two_view}")
print(f"Files in output folder: {total_output_files}")

