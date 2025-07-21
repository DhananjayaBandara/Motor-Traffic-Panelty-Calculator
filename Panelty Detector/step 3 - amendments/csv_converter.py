import pandas as pd
import chardet
import os

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def convert_csv_to_utf8(input_file_path, output_file_path=None):
    """Convert CSV file to UTF-8 encoding"""
    
    if output_file_path is None:
        # Create output filename by adding '_utf8' before the extension
        base_name = os.path.splitext(input_file_path)[0]
        extension = os.path.splitext(input_file_path)[1]
        output_file_path = f"{base_name}_utf8{extension}"
    
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    
    # Detect encoding
    print("Detecting file encoding...")
    detected_encoding = detect_encoding(input_file_path)
    print(f"Detected encoding: {detected_encoding}")
    
    # Try multiple encodings if detection fails
    encodings_to_try = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    df = None
    successful_encoding = None
    
    for encoding in encodings_to_try:
        if encoding is None:
            continue
        try:
            print(f"Trying to read with encoding: {encoding}")
            df = pd.read_csv(input_file_path, encoding=encoding)
            successful_encoding = encoding
            print(f"Successfully read file with encoding: {encoding}")
            break
        except Exception as e:
            print(f"Failed with encoding {encoding}: {str(e)}")
            continue
    
    if df is None:
        print("Failed to read the file with any encoding. Trying with error handling...")
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8', errors='ignore')
            successful_encoding = 'utf-8 (with errors ignored)'
            print("Successfully read file with UTF-8 (ignoring errors)")
        except Exception as e:
            print(f"Final attempt failed: {str(e)}")
            return False
    
    # Clean the data
    print("Cleaning data...")
    
    # Replace problematic characters and clean text
    for column in df.columns:
        if df[column].dtype == 'object':  # Text columns
            df[column] = df[column].astype(str).apply(lambda x: 
                x.replace('\ufffd', '')  # Remove replacement characters
                 .replace('\x00', '')    # Remove null bytes
                 .replace('\r\n', ' ')   # Replace line breaks with spaces
                 .replace('\n', ' ')     # Replace line breaks with spaces
                 .replace('\r', ' ')     # Replace line breaks with spaces
                 .strip()                # Remove leading/trailing whitespace
            )
    
    # Forward fill c1 and c2 columns (like Excel drag down)
    print("Forward filling c1 and c2 columns...")
    if 'c1' in df.columns:
        # Replace 'nan' strings and empty strings with NaN, then forward fill
        df['c1'] = df['c1'].replace(['nan', '', ' '], pd.NA)
        df['c1'] = df['c1'].fillna(method='ffill')
        print(f"Forward filled c1 column")
    
    if 'c2' in df.columns:
        # Replace 'nan' strings and empty strings with NaN, then forward fill
        df['c2'] = df['c2'].replace(['nan', '', ' '], pd.NA)
        df['c2'] = df['c2'].fillna(method='ffill')
        print(f"Forward filled c2 column")
    
    # Handle c4 and c5 columns based on c5 values
    print("Processing c4 and c5 columns...")
    if 'c4' in df.columns and 'c5' in df.columns:
        # Clean c5 column first
        df['c5'] = df['c5'].replace(['nan', '', ' '], pd.NA)
        df['c4'] = df['c4'].replace(['nan', '', ' '], pd.NA)
        
        for i in range(len(df)):
            # If c5 is NaN, keep c4 as it is
            if pd.isna(df.loc[i, 'c5']):
                # c4 stays the same (no change needed)
                pass
            else:
                # If c5 has a value, replace c4 with the value from the upper row
                if i > 0:  # Make sure we're not at the first row
                    df.loc[i, 'c4'] = df.loc[i-1, 'c4']
        
        print("Processed c4 and c5 columns based on c5 values")
    
    # Handle c3 and c4 columns based on c4 values
    print("Processing c3 and c4 columns...")
    if 'c3' in df.columns and 'c4' in df.columns:
        # Clean c3 column
        df['c3'] = df['c3'].replace(['nan', '', ' '], pd.NA)
        
        for i in range(len(df)):
            # If c4 is NaN, keep c3 as it is
            if pd.isna(df.loc[i, 'c4']):
                # c3 stays the same (no change needed)
                pass
            else:
                # If c4 has a value, replace c3 with the value from the upper row
                if i > 0:  # Make sure we're not at the first row
                    df.loc[i, 'c3'] = df.loc[i-1, 'c3']
        
        print("Processed c3 and c4 columns based on c4 values")
    
    # Save as UTF-8
    print("Saving as UTF-8...")
    try:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"Successfully converted and saved to: {output_file_path}")
        
        # Verify the conversion
        print("Verifying conversion...")
        test_df = pd.read_csv(output_file_path, encoding='utf-8')
        print(f"Verification successful! File has {len(test_df)} rows and {len(test_df.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"Failed to save file: {str(e)}")
        return False

def main():
    """Main function to convert the CSV file"""
    input_file = "motor_traffic_act.csv"
    output_file = "motor_traffic_act_utf8.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    print("=== CSV Encoding Converter ===")
    print("Converting Motor Traffic Act CSV to UTF-8 encoding...")
    print()
    
    success = convert_csv_to_utf8(input_file, output_file)
    
    if success:
        print()
        print("=== Conversion Complete ===")
        print(f"Your converted file is ready: {output_file}")
        print("You can now use this file with the RAG.py script.")
    else:
        print()
        print("=== Conversion Failed ===")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
