#!/usr/bin/env python3
"""
Final script to combine CNN predictions back into words and cells
Creates both CSV output and visualization of the reconstructed table
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2
from pathlib import Path

def parse_filename(filename):
    """
    Parse filename to extract cell coordinates and character information
    Handles multiple formats:
    - cell_r{row}_c{col}_blob_{blob_num}_word_{word_num}_char_{char_idx}.png
    - cell_r{row}_c{col}_blob_{blob1}_blob_{blob2}_word_{word_num}_char_{char_idx}.png
    """
    # Try the more complex pattern first (with double blob)
    pattern1 = r'cell_r(\d+)_c(\d+)_blob_(\d+)_blob_(\d+)_word_(\d+)_char_(\d+)\.png'
    match = re.match(pattern1, filename)

    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        blob_num1 = int(match.group(3))
        blob_num2 = int(match.group(4))
        word_num = int(match.group(5))
        char_idx = int(match.group(6))
        return row, col, f"{blob_num1}_{blob_num2}", word_num, char_idx

    # Try the simpler pattern
    pattern2 = r'cell_r(\d+)_c(\d+)_blob_(\d+)_word_(\d+)_char_(\d+)\.png'
    match = re.match(pattern2, filename)

    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        blob_num = int(match.group(3))
        word_num = int(match.group(4))
        char_idx = int(match.group(5))
        return row, col, blob_num, word_num, char_idx
    else:
        print(f"Warning: Could not parse filename: {filename}")
        return None

def load_predictions():
    """Load CNN predictions from CSV file"""
    predictions_file = 'experiment/predictions.csv'
    
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df)} predictions from {predictions_file}")
    
    return df

def get_all_cells():
    """Get all cell coordinates from cells_cleaned directory"""
    cells_dir = 'cells_cleaned'
    
    if not os.path.exists(cells_dir):
        raise FileNotFoundError(f"Cells directory not found: {cells_dir}")
    
    cell_pattern = r'cell_r(\d+)_c(\d+)\.png'
    cells = set()
    
    for filename in os.listdir(cells_dir):
        match = re.match(cell_pattern, filename)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            cells.add((row, col))
    
    print(f"Found {len(cells)} cells in {cells_dir}")
    return cells

def reconstruct_words_and_cells(predictions_df):
    """
    Reconstruct words from character predictions and organize by cells
    """
    # Dictionary to store cell contents
    # Structure: {(row, col): {word_num: reconstructed_word}}
    cell_contents = defaultdict(lambda: defaultdict(list))
    
    # Process each prediction
    for _, row in predictions_df.iterrows():
        filename = row['filename']
        predicted_digit = str(int(row['predicted_class']))
        confidence = row.get('confidence', 1.0)
        
        # Parse filename
        parsed = parse_filename(filename)
        if parsed is None:
            continue
            
        cell_row, cell_col, blob_num, word_num, char_idx = parsed
        
        # Store character with its position
        cell_contents[(cell_row, cell_col)][word_num].append({
            'char_idx': char_idx,
            'digit': predicted_digit,
            'confidence': confidence
        })
    
    # Sort characters within each word and reconstruct
    reconstructed_cells = {}
    
    for (cell_row, cell_col), words in cell_contents.items():
        cell_words = []
        
        for word_num in sorted(words.keys()):
            # Sort characters by their index
            chars = sorted(words[word_num], key=lambda x: x['char_idx'])
            
            # Reconstruct word
            word = ''.join([char['digit'] for char in chars])
            avg_confidence = np.mean([char['confidence'] for char in chars])
            
            cell_words.append({
                'word': word,
                'confidence': avg_confidence,
                'word_num': word_num
            })
        
        # Sort words by word number
        cell_words.sort(key=lambda x: x['word_num'])
        
        # Combine words with spaces
        if cell_words:
            full_text = ' '.join([w['word'] for w in cell_words])
            avg_confidence = np.mean([w['confidence'] for w in cell_words])
        else:
            full_text = ''
            avg_confidence = 0.0
        
        reconstructed_cells[(cell_row, cell_col)] = {
            'text': full_text,
            'confidence': avg_confidence,
            'word_count': len(cell_words)
        }
    
    return reconstructed_cells

def create_table_csv(reconstructed_cells, all_cells, output_file='data/csv/reconstructed_table.csv'):
    """
    Create CSV file with all cells, using 'WORD' for non-numeric cells
    """
    # Find table dimensions
    if not all_cells:
        print("No cells found!")
        return
    
    # Use fixed dimensions based on cells_cleaned directory (21 rows × 20 columns)
    max_row = 20  # rows 0-20
    max_col = 19  # columns 0-19

    print(f"Table dimensions: {max_row + 1} rows x {max_col + 1} columns")

    # Create table data
    table_data = []

    for row in range(max_row + 1):
        row_data = []
        for col in range(max_col + 1):
            if (row, col) in reconstructed_cells:
                # Cell has numeric predictions
                cell_info = reconstructed_cells[(row, col)]
                content = cell_info['text'] if cell_info['text'] else 'WORD'
            elif (row, col) in all_cells:
                # Cell exists but no numeric predictions
                content = 'WORD'
            else:
                # Cell doesn't exist
                content = ''
            
            row_data.append(content)
        table_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    df.columns = [f'col_{i}' for i in range(len(df.columns))]
    df.index = [f'row_{i}' for i in range(len(df))]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=True)
    print(f"Table CSV saved to: {output_file}")
    
    return df

def create_visualization(csv_file='data/csv/reconstructed_table.csv', output_file='results/table_visualization.png'):
    """
    Create visualization of the reconstructed table by reading from CSV file
    """
    # Read the CSV file
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return

    # Load the table data
    table_df = pd.read_csv(csv_file, index_col=0)
    print(f"Loaded table with {len(table_df)} rows and {len(table_df.columns)} columns")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

    # Plot 1: Table heatmap showing cell types
    table_numeric = table_df.copy()

    # Create numeric representation for visualization
    # 0 = empty, 1 = WORD, 2 = numbers
    for i in range(len(table_numeric)):
        for j in range(len(table_numeric.columns)):
            cell_value = table_df.iloc[i, j]
            if pd.isna(cell_value) or cell_value == '':
                table_numeric.iloc[i, j] = 0
            elif cell_value == 'WORD':
                table_numeric.iloc[i, j] = 1
            else:
                table_numeric.iloc[i, j] = 2

    # Convert to numeric
    table_numeric = table_numeric.astype(float)

    # Create heatmap
    sns.heatmap(table_numeric,
                annot=False,
                cmap=['white', 'lightblue', 'orange'],
                cbar_kws={'label': 'Cell Type'},
                ax=ax1)

    ax1.set_title('Table Structure Overview\n(White=Empty, Blue=Text, Orange=Numbers)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')

    # Plot 2: Detailed view with actual content (sample)
    # Show a subset of the table with actual text
    sample_size = max(15, len(table_df))
    sample_cols = max(10, len(table_df.columns))

    sample_df = table_df.iloc[:sample_size, :sample_cols]

    # Create a matrix for coloring
    color_matrix = np.zeros((len(sample_df), len(sample_df.columns)))
    annotations = []

    for i in range(len(sample_df)):
        row_annotations = []
        for j in range(len(sample_df.columns)):
            cell_value = sample_df.iloc[i, j]
            if pd.isna(cell_value) or cell_value == '':
                color_matrix[i, j] = 0
                row_annotations.append('')
            elif cell_value == 'WORD':
                color_matrix[i, j] = 1
                row_annotations.append('WORD')
            else:
                color_matrix[i, j] = 2
                # Truncate long numbers for display
                display_text = str(cell_value)[:8] + '...' if len(str(cell_value)) > 8 else str(cell_value)
                row_annotations.append(display_text)
        annotations.append(row_annotations)

    # Create heatmap with annotations
    sns.heatmap(color_matrix,
                annot=annotations,
                fmt='',
                cmap=['white', 'lightblue', 'orange'],
                cbar_kws={'label': 'Cell Type'},
                ax=ax2,
                annot_kws={'size': 8})

    ax2.set_title(f'Table Content Sample (First {sample_size} rows, {sample_cols} columns)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Rows')

    plt.tight_layout()

    # Save visualization
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

    plt.close()

    # Print some statistics about the table
    total_cells = len(table_df) * len(table_df.columns)
    empty_cells = 0
    word_cells = 0
    number_cells = 0

    for i in range(len(table_df)):
        for j in range(len(table_df.columns)):
            cell_value = table_df.iloc[i, j]
            if pd.isna(cell_value) or cell_value == '':
                empty_cells += 1
            elif cell_value == 'WORD':
                word_cells += 1
            else:
                number_cells += 1

    print(f"\nTable Statistics:")
    print(f"  Total cells: {total_cells}")
    print(f"  Empty cells: {empty_cells} ({empty_cells/total_cells*100:.1f}%)")
    print(f"  Text cells (WORD): {word_cells} ({word_cells/total_cells*100:.1f}%)")
    print(f"  Number cells: {number_cells} ({number_cells/total_cells*100:.1f}%)")

def create_statistics_report(reconstructed_cells, all_cells):
    """
    Create a statistics report about the reconstruction
    """
    print("\n" + "="*60)
    print("RECONSTRUCTION STATISTICS")
    print("="*60)
    
    total_cells = len(all_cells)
    numeric_cells = len(reconstructed_cells)
    word_cells = total_cells - numeric_cells
    
    print(f"Total cells found: {total_cells}")
    print(f"Cells with numeric content: {numeric_cells}")
    print(f"Cells with text content (WORD): {word_cells}")
    print(f"Numeric content percentage: {numeric_cells/total_cells*100:.1f}%")
    
    if reconstructed_cells:
        # Confidence statistics
        confidences = [cell['confidence'] for cell in reconstructed_cells.values()]
        print(f"\nConfidence Statistics:")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
        
        # Word count statistics
        word_counts = [cell['word_count'] for cell in reconstructed_cells.values()]
        print(f"\nWord Count Statistics:")
        print(f"  Average words per numeric cell: {np.mean(word_counts):.1f}")
        print(f"  Max words in a cell: {np.max(word_counts)}")
        
        # Character length statistics
        text_lengths = [len(cell['text']) for cell in reconstructed_cells.values()]
        print(f"\nText Length Statistics:")
        print(f"  Average characters per numeric cell: {np.mean(text_lengths):.1f}")
        print(f"  Max characters in a cell: {np.max(text_lengths)}")
    
    print("="*60)

def main():
    """Main function to orchestrate the reconstruction process"""
    print("Starting table reconstruction from CNN predictions...")
    
    try:
        # Load predictions
        # predictions_df = load_predictions()
        
        # Get all cells
        # all_cells = get_all_cells()
        
        # Reconstruct words and cells
        print("\nReconstructing words from character predictions...")
        # reconstructed_cells = reconstruct_words_and_cells(predictions_df)
        
        # Create CSV output
        print("\nCreating CSV output...")
        # table_df = create_table_csv(reconstructed_cells, all_cells)
        
        # Create visualization
        print("\nCreating visualization...")
        create_visualization()  # Now reads from CSV file directly
        
        print(f"\n✅ Reconstruction complete!")
        print(f"📄 CSV file: data/csv/reconstructed_table.csv")
        print(f"🖼️  Visualization: results/table_visualization.png")
        
    except Exception as e:
        print(f"❌ Error during reconstruction: {e}")
        raise

def visualize_table_only():
    """
    Standalone function to create visualization from existing CSV file
    """
    print("Creating visualization from existing CSV file...")
    create_visualization()
    print("✅ Visualization complete!")

if __name__ == "__main__":
    # Check if we want to run just visualization
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize-only":
        visualize_table_only()
    else:
        main()
