"""
Pipeline runner for OCR processing.
Simplified version that works with Django's threading model.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional


class PipelineRunner:
    """Runs the OCR pipeline with progress callbacks."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Default to parent of webapp directory
            self.base_dir = Path(__file__).parent.parent.parent.absolute()
        else:
            self.base_dir = Path(base_dir).absolute()

        self.total_steps = 14
        self.current_step = 0

    def get_progress_percentage(self) -> int:
        """Calculate current progress as percentage."""
        return int((self.current_step / self.total_steps) * 100)

    def run_step(
        self,
        description: str,
        command: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """Run a pipeline step with progress tracking."""
        self.current_step += 1
        progress = self.get_progress_percentage()

        if progress_callback:
            progress_callback(progress, description)
        else:
            print(f"[{progress}%] {description}")

        # Change to base directory
        original_dir = os.getcwd()
        os.chdir(self.base_dir)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per step
            )

            os.chdir(original_dir)

            if result.returncode != 0:
                error_msg = f"Step failed: {description}"
                if result.stderr:
                    error_msg += f"\n{result.stderr[:500]}"  # First 500 chars
                if progress_callback:
                    progress_callback(progress, error_msg)
                else:
                    print(error_msg)
                return False

            return True

        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            error_msg = f"Step timed out: {description}"
            if progress_callback:
                progress_callback(progress, error_msg)
            return False
        except Exception as e:
            os.chdir(original_dir)
            error_msg = f"Exception: {description} - {str(e)}"
            if progress_callback:
                progress_callback(progress, error_msg)
            return False

    def run_pipeline(
        self,
        image_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Run the complete pipeline.

        Returns:
            (success, csv_path, image_path)
        """
        self.current_step = 0

        if not os.path.exists(image_path):
            if progress_callback:
                progress_callback(0, f"Error: file not found: {image_path}")
            return False, None, None

        if progress_callback:
            progress_callback(0, "Starting OCR pipeline...")

        # Define all pipeline steps
        steps = [
            ("Preprocessing table and extracting cells",
             f"python utils/preprocessing.py --input '{image_path}'"),

            ("Removing black borders from cells",
             "python src/clean_white_words.py --mode remove_border --folder cells_production"),

            ("Cleaning white/empty cells",
             "python src/clean_white_words.py --mode clean_cells --folder cells_cleaned"),

            ("Extracting blobs from cells",
             "python src/try_blobs.py --mode blobs"),

            ("Cleaning white/empty blobs",
             "python src/clean_white_words.py --mode clean_cells --folder blobs"),

            ("Extracting words from blobs",
             "python src/try_blobs.py --mode words"),

            ("Resizing words to standard dimensions",
             "python src/resize.py"),

            ("Improving quality of word images",
             "python experiment/improve_quality.py"),

            ("Segmenting words into characters",
             "python src/seg_cells.py"),

            ("Sorting numbers CSV",
             "python src/sort_csv.py --file data/csv/numbers_latest.csv"),

            ("Removing invalid character images",
             "python src/find_size.py"),

            ("Sorting numbers CSV again",
             "python src/sort_csv.py --file data/csv/numbers_latest.csv"),

            ("Predicting digits using CNN",
             "python src/test_digits_model.py"),

            ("Sorting predictions",
             "python src/sort_csv.py --file experiment/predictions.csv"),
        ]

        # Run all steps
        for description, command in steps:
            if not self.run_step(description, command, progress_callback):
                return False, None, None

        # Final step: combine predictions
        combine_script = self.base_dir / "src" / "combine_predictions_to_table.py"
        if combine_script.exists():
            if not self.run_step(
                "Combining predictions into final table",
                f"python {combine_script}",
                progress_callback
            ):
                return False, None, None

        if progress_callback:
            progress_callback(100, "Pipeline completed successfully!")

        # Return output file paths
        csv_path = self.base_dir / "data" / "output" / "reconstructed_table.csv"
        image_path = self.base_dir / "data" / "output" / "table_visualization.png"

        if csv_path.exists() and image_path.exists():
            return True, str(csv_path), str(image_path)
        else:
            return False, None, None
