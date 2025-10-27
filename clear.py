import os

def clear_directory(folder_path):
    """
    Deletes all files (not subfolders) inside the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        return
    
    count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            
            if os.path.isfile(file_path) and file_path.endswith(".png"):
                os.remove(file_path)
                count += 1
        except Exception as e:
            print(f"⚠️ Could not delete {file_path}: {e}")
    
    print(f"✅ Deleted {count} files from: {folder_path}")

# Example usage:
if __name__ == "__main__":
    # folder = "data/synth_data"  # change to your folder
    # clear_directory(folder)
    
    # folder_2 = "steps_out"  # change to your folder
    # clear_directory(folder_2)
    
    # import pathlib
    # main_folder = "data/synth_data"
    # base_path = pathlib.Path(main_folder)
    
    # folder = "synthetic_data/images"
    
    # # clear_directory(folder)
    # folder = "blobs"
    
    # clear_directory(folder)
    # folder = "cells_out"
    # clear_directory(folder)
    # folder = "characters"
    
    # clear_directory(folder)
    folder = "experiment/debug_out"
    clear_directory(folder)
    # folder = "experiment"
    # clear_directory(folder)
    folder = "results"
    
    clear_directory(folder)
    folder = "generated_chars"
    clear_directory(folder)
    # folder = ""
    # folder = "synthetic_data/images"
    # clear_directory(folder)
    # folder = "synthetic_data/words"
    # clear_directory(folder)
    
    # clear_directory(folder)
    # r1 c4  c5 
    # r2 c0
    # r2 c3 c5 c9
    
    # print(f"Subdirectories found in: {base_path}")
    # for item in base_path.iterdir():
    #     if item.is_dir():
    #         clear_directory(item)
    #         # delete the folder - item itself
    #         try:
    #             os.rmdir(item)
    #             print(f"✅ Deleted folder: {item}")
    #         except Exception as e:
    #             print(f"⚠️ Could not delete folder {item}: {e}")

