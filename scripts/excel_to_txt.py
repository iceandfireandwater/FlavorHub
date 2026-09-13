"""
Convert Excel file to text format for Milvus ingestion.
Each row becomes a paragraph, sheets are separated by empty lines.
"""

import pandas as pd
from pathlib import Path
import sys
from typing import Optional


def excel_to_txt(excel_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert Excel file to txt format suitable for Milvus ingestion.
    
    Args:
        excel_path: Path to the Excel file
        output_path: Optional output txt path (default: same name with .txt extension)
    
    Returns:
        Path to the generated txt file
    """
    excel_file = Path(excel_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    if output_path is None:
        output_path = excel_file.with_suffix('.txt')
    else:
        output_path = Path(output_path)
    
    # Read all sheets
    xls = pd.ExcelFile(excel_file)
    paragraphs = []
    
    for sheet_name in xls.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Skip empty sheets
        if df.empty:
            continue
        
        # Convert each row to a text paragraph
        for _, row in df.iterrows():
            # Filter out NaN values and convert to string
            row_values = [str(v) for v in row.values if pd.notna(v)]
            if row_values:
                # Join non-empty values with separators
                paragraph = ' | '.join(row_values)
                if paragraph.strip():
                    paragraphs.append(paragraph)
        
        # Add empty line between sheets
        paragraphs.append('')
    
    # Write to txt file (paragraphs separated by double newlines)
    output_path.write_text('\n\n'.join(paragraphs), encoding='utf-8')
    print(f"Converted {len(paragraphs)} paragraphs to: {output_path}")
    return str(output_path)


def main():
    # Default paths
    excel_path = Path(__file__).parent.parent / "kb_ingest" / "data" / "历史菜谱源头.xlsx"
    output_path = Path(__file__).parent.parent / "data" / "kb" / "历史菜谱源头.txt"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    result = excel_to_txt(str(excel_path), str(output_path))
    print(f"Output saved to: {result}")


if __name__ == "__main__":
    main()
