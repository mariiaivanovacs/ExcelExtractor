"""
Main pipeline to run OCR on low-quality table images.

Steps:
1. Preprocess table image and extract cells
2. Extract blobs from cells
3. Extract words from blobs
4. Segment words into characters
5. Clean invalid character images
6. Classify words into NUMBER / OTHER
7. Save final results to CSV


"""


# IMPORTANT NOTE: STEP 6 IS NOT INCLUDED IN CURRENT PIPELINE. 
# IF YOU RUN FIRST TIME, UNCOMMENT !!!!!!


import os
import sys
import subprocess
import time


def run_step(description, command):
    """Run a shell command and print progress."""
    print(f"\n🔹 {description}")
    start = time.time()
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Step failed: {description}")
        sys.exit(1)
    print(f"✅ Done: {description} in {time.time() - start:.2f}s")


def main(image_path):
    print("IMPORTANT NOTE: STEP 6 IS NOT INCLUDED IN CURRENT PIPELINE. \nIF YOU RUN FIRST TIME, UNCOMMENT STEP 6 IN run_pipeline.py FILE")
    user_reply = input("Do you want to proceed? (y/n): ")
    if user_reply.lower() != "y":
        print("Exiting...")
        sys.exit(1)
        
    # === 0. Check input ===
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    print(f"\n🚀 Starting OCR pipeline for: {image_path}\n")

    # === 1. Preprocess and extract cells ===
    run_step(
        "Preprocessing table and extracting cells",
        f"python utils/preprocessing.py --input '{image_path}'"
    )
    
    # === 1.1. Clean cells ===
    run_step(
        "Removing black borders from cells",
        "python src/clean_white_words.py --mode remove_border --folder cells_production"
    )

    run_step(
        "Cleaning white/empty cells",
        "python src/clean_white_words.py --mode clean_cells --folder cells_cleaned"
    )

    # === 2. From cells → blobs ===
    run_step(
        "Extracting blobs from cells",
        "python src/try_blobs.py --mode blobs"
    )
    run_step(
        "Cleaning white/empty blobs",
        "python src/clean_white_words.py --mode clean_cells --folder blobs"
    )

    # === 3. From blobs → words ===
    run_step(
        "Extracting words from blobs",
        "python src/try_blobs.py --mode words"
    )

    # === 3.1. Resize and improve words ===
    run_step(
        "Resizing words to standard dimensions",
        "python src/resize.py"
    )

    run_step(
        "Improving quality of word images",
        "python experiment/improve_quality.py"
    )
    
    
    
    # === 4. From words → characters (32x32) ===
    run_step(
        "Segmenting words into individual characters",
        "python src/seg_cells.py"
    )
    
    
    # # === 6. Classify NUMBER vs OTHER ===
    # run_step(
    #     "Classifying words as NUMBER or OTHER",
    #     "python experiment/enhanced_cell_type_detector.py"
    # )
    
    run_step(
        "Sort numbers_latest.csv",
        "python src/sort_csv.py --file data/csv/numbers_latest.csv"
    )

    # === 5. Clean and filter characters ===
    run_step(
        "Removing invalid (white/empty) character images",
        "python src/find_size.py"
    )

    run_step(
        "Sorting numbers CSV file",
        "python src/sort_csv.py --file data/csv/numbers_latest.csv"
    )

    # === 6. Classify digits (0-9) ===
    run_step(
        "Predicting digits using CNN classifier",
        "python src/test_digits_model.py"
    )

    run_step(
        "Sorting predictions CSV file",
        "python src/sort_csv.py --file experiment/predictions.csv"
    )
    

    # === 7. Combine predictions into final table ===
    combine_script = "src/combine_predictions_to_table.py"
    if os.path.exists(combine_script):
        run_step(
            "Combining predictions into final table CSV",
            f"python {combine_script}"
        )
    else:
        print("⚠️ combine_predictions_to_table.py not found — skipping final CSV merge.")

    print("\n🎯 Pipeline completed successfully!")
    print("Check output files in: results/ or steps_out/ directories.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <path_to_input_table_image>")
        sys.exit(1)

    image_file = sys.argv[1]
    main(image_file)


# Example of command: 
# python3 run_pipeline.py data/input/original.jpeg 