#!/usr/bin/env python3
"""
Check Model Input Requirements for Student Dropout Prediction

This script shows exactly what input fields your dropout prediction model needs
and provides sample data in the correct format.
"""

import pandas as pd
import json

def check_model_requirements():
    """Check what input fields the model needs and show sample data."""
    
    print("=" * 70)
    print("STUDENT DROPOUT PREDICTION MODEL - INPUT REQUIREMENTS")
    print("=" * 70)
    
    # Check Vietnamese dataset (this matches your model)
    print("\n📊 VIETNAMESE DATASET (student_data_VN.csv)")
    print("-" * 50)
    
    try:
        df_vn = pd.read_csv('student_data_VN.csv')
        
        # Show all columns
        print(f"Total columns: {len(df_vn.columns)}")
        print("All columns:")
        for i, col in enumerate(df_vn.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Show required features (excluding student_id and Target)
        required_features = [col for col in df_vn.columns if col not in ['student_id', 'Target']]
        print(f"\n✅ REQUIRED INPUT FEATURES ({len(required_features)} features):")
        print("(These are the features your model expects)")
        for i, col in enumerate(required_features, 1):
            col_type = str(df_vn[col].dtype)
            sample_val = df_vn[col].iloc[0] if len(df_vn) > 0 else "N/A"
            print(f"  {i:2d}. {col:<35} | Type: {col_type:<10} | Sample: {sample_val}")
        
        # Show sample data in correct format
        print(f"\n📝 SAMPLE DATA FOR MODEL INPUT:")
        print("(This is the format your API expects)")
        
        sample_records = df_vn[required_features].head(3).to_dict('records')
        
        api_input = {
            "data": sample_records
        }
        
        print(json.dumps(api_input, indent=2))
        
        # Save sample data
        with open('model_input_sample.json', 'w', encoding='utf-8') as f:
            json.dump(api_input, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Sample data saved to: model_input_sample.json")
        
    except FileNotFoundError:
        print("❌ student_data_VN.csv not found!")
        return
    
    # Check Portuguese dataset for comparison
    print(f"\n\n📊 PORTUGUESE DATASET (student_dropout_dataset_Portugal.csv)")
    print("-" * 50)
    
    try:
        df_pt = pd.read_csv('student_dropout_dataset_Portugal.csv')
        
        print(f"Total columns: {len(df_pt.columns)}")
        print("⚠️  NOTE: This dataset has different features than your Vietnamese model!")
        print("    You would need to map/transform these features to use with your current model.")
        
        # Show first few columns as examples
        print(f"\nFirst 10 columns (out of {len(df_pt.columns)}):")
        for i, col in enumerate(df_pt.columns[:10], 1):
            print(f"  {i:2d}. {col}")
        
        if len(df_pt.columns) > 10:
            print(f"  ... and {len(df_pt.columns) - 10} more columns")
            
    except FileNotFoundError:
        print("❌ student_dropout_dataset_Portugal.csv not found!")
    
    # Summary and next steps
    print(f"\n\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    
    print(f"\n🎯 YOUR MODEL EXPECTS THESE {len(required_features)} FEATURES:")
    for i, col in enumerate(required_features, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📋 TO USE YOUR MODEL:")
    print("1. Prepare data with exactly these features (in any order)")
    print("2. Format as JSON with structure: {\"data\": [record1, record2, ...]}")
    print("3. Send to your API endpoint: /predict")
    
    print(f"\n💡 TIP: Use the generated 'model_input_sample.json' as a template!")
    
    print(f"\n🔍 FEATURE DESCRIPTIONS:")
    print("  - Gender: 0=Female, 1=Male")
    print("  - Curricular units: Number of courses enrolled/approved")
    print("  - Grades: Average grades (0.0 if no courses)")
    print("  - Debtor: 0=No debt, 1=Has debt")
    print("  - Tuition fees: 0=Not up to date, 1=Up to date")
    print("  - Total counts: Sum of enrolled/approved/failed courses")
    print("  - Average grade: Overall academic performance")
    print("  - Unpassed courses: Number of failed courses")

if __name__ == "__main__":
    check_model_requirements()
