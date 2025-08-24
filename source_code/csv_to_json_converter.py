#!/usr/bin/env python3
"""
CSV to JSON Converter for Student Dropout Prediction Model

This script converts CSV files to JSON format and shows the required input fields
for the dropout prediction model. It handles both Vietnamese and Portuguese datasets.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any

def convert_csv_to_json(csv_file_path: str, output_file: str = None, num_samples: int = 5) -> Dict[str, Any]:
    """
    Convert CSV file to JSON format and show model input requirements.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Output JSON file path (optional)
        num_samples: Number of sample records to include in output
    
    Returns:
        Dictionary containing conversion results and model requirements
    """
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Display basic information
    print(f"\nDataset Information:")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if this is Vietnamese or Portuguese dataset
    is_vietnamese = 'student_id' in df.columns
    dataset_type = "Vietnamese" if is_vietnamese else "Portuguese"
    print(f"\nDataset type: {dataset_type}")
    
    # Show required input fields for the model
    if is_vietnamese:
        # For Vietnamese dataset, exclude student_id and Target columns
        required_columns = [col for col in df.columns if col not in ['student_id', 'Target']]
        print(f"\nRequired input fields for Vietnamese model ({len(required_columns)} features):")
        for i, col in enumerate(required_columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Create sample data for model input
        sample_data = df[required_columns].head(num_samples).to_dict('records')
        
    else:
        # For Portuguese dataset, show all columns (you may need to map these)
        required_columns = df.columns.tolist()
        print(f"\nAll columns in Portuguese dataset ({len(required_columns)} features):")
        for i, col in enumerate(required_columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Create sample data
        sample_data = df.head(num_samples).to_dict('records')
    
    # Create the output structure
    result = {
        "dataset_info": {
            "file_path": csv_file_path,
            "dataset_type": dataset_type,
            "total_records": len(df),
            "total_features": len(required_columns),
            "required_features": required_columns
        },
        "sample_data": sample_data,
        "model_input_format": {
            "description": "Format required by the dropout prediction API",
            "example": {
                "data": sample_data[:2]  # Show first 2 samples as example
            }
        }
    }
    
    # Save to JSON file if output path is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output saved to: {output_file}")
    
    return result

def create_model_input_template(csv_file_path: str, output_file: str = None) -> Dict[str, Any]:
    """
    Create a template JSON file that can be used directly with the prediction API.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Output template file path (optional)
    
    Returns:
        Dictionary containing the template
    """
    
    df = pd.read_csv(csv_file_path)
    is_vietnamese = 'student_id' in df.columns
    
    if is_vietnamese:
        # For Vietnamese dataset, create template without student_id and Target
        required_columns = [col for col in df.columns if col not in ['student_id', 'Target']]
        
        # Get data types and sample values
        column_info = {}
        for col in required_columns:
            col_type = str(df[col].dtype)
            sample_value = df[col].iloc[0] if len(df) > 0 else None
            column_info[col] = {
                "type": col_type,
                "sample_value": sample_value,
                "description": f"Feature: {col}"
            }
        
        template = {
            "description": "Template for Vietnamese Student Dropout Prediction",
            "required_features": required_columns,
            "feature_info": column_info,
            "api_input_format": {
                "data": [
                    {col: "VALUE_HERE" for col in required_columns}
                ]
            },
            "example_with_real_data": {
                "data": df[required_columns].head(3).to_dict('records')
            }
        }
        
    else:
        # For Portuguese dataset, show all columns
        required_columns = df.columns.tolist()
        
        column_info = {}
        for col in required_columns:
            col_type = str(df[col].dtype)
            sample_value = df[col].iloc[0] if len(df) > 0 else None
            column_info[col] = {
                "type": col_type,
                "sample_value": sample_value,
                "description": f"Feature: {col}"
            }
        
        template = {
            "description": "Template for Portuguese Student Dataset",
            "note": "This dataset has different features than the Vietnamese model. You may need to map or transform these features.",
            "all_features": required_columns,
            "feature_info": column_info,
            "example_data": df.head(3).to_dict('records')
        }
    
    # Save template if output path is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"Template saved to: {output_file}")
    
    return template

def main():
    """Main function to run the converter."""
    
    print("=" * 60)
    print("CSV to JSON Converter for Student Dropout Prediction")
    print("=" * 60)
    
    # Check available CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory!")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file}")
        print(f"{'='*60}")
        
        try:
            # Convert to JSON
            json_output_file = csv_file.replace('.csv', '_converted.json')
            result = convert_csv_to_json(csv_file, json_output_file)
            
            # Create template
            template_file = csv_file.replace('.csv', '_template.json')
            template = create_model_input_template(csv_file, template_file)
            
            print(f"\n✓ Successfully processed {csv_file}")
            print(f"  - JSON conversion: {json_output_file}")
            print(f"  - Template file: {template_file}")
            
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Conversion completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the generated JSON files to see the data structure")
    print("2. Use the template files to understand required input format")
    print("3. For Vietnamese dataset: Use the converted data directly with your API")
    print("4. For Portuguese dataset: You may need to map features to match the model")

if __name__ == "__main__":
    main()
