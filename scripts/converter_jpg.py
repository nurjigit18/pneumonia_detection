import os
import shutil
import argparse

def convert_jpeg_to_jpg(input_dir, output_dir=None):
    """
    Convert all .jpeg files in the input directory to .jpg files in the output directory.
    If output_dir is not provided, files will be converted in place.
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Count conversions for reporting
    converted_count = 0
    
    # Walk through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Check if the file has .jpeg extension (case insensitive)
            if file.lower().endswith('.jpg'):
                # Get the full path of the original file
                original_file = os.path.join(root, file)
                
                # Create new filename by replacing .jpeg with .jpg
                new_name = os.path.splitext(file)[0] + '.jpeg'
                
                if output_dir:
                    # If output directory specified, place the file there
                    relative_path = os.path.relpath(root, input_dir)
                    new_dir = os.path.join(output_dir, relative_path)
                    
                    # Create subdirectory structure if needed
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    
                    new_file = os.path.join(new_dir, new_name)
                else:
                    # Otherwise, place the file in the same directory
                    new_file = os.path.join(root, new_name)
                
                # Copy the file with the new extension
                shutil.copy2(original_file, new_file)
                print(f"Converted: {original_file} -> {new_file}")
                converted_count += 1
    
    return converted_count

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert .jpeg files to .jpg')
    parser.add_argument('input_dir', help='Directory containing .jpeg files')
    parser.add_argument('--output-dir', '-o', help='Output directory for .jpg files (optional)')
    parser.add_argument('--delete-originals', '-d', action='store_true', 
                        help='Delete original .jpeg files after conversion')
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Starting conversion from directory: {args.input_dir}")
    
    # Convert files
    converted = convert_jpeg_to_jpg(args.input_dir, args.output_dir)
    
    # Delete original files if requested
    if args.delete_originals and converted > 0:
        print("Deleting original .jpeg files...")
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.lower().endswith('.jpeg'):
                    os.remove(os.path.join(root, file))
                    print(f"Deleted: {os.path.join(root, file)}")
    
    print(f"Conversion complete. Converted {converted} files.")

if __name__ == "__main__":
    main()