

import os
import argparse
import glob
import re

def extract_functions_and_classes(file_path):
   
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        snippets = []
        # Extract functions (look for indentation)
        function_pattern = re.compile(r'def\s+\w+\s*\(.*?\).*?:(.*?)(?=\n\S|\Z)', re.DOTALL)
        for match in function_pattern.finditer(content):
            func_def = match.group(0)
            # Check if this is a method in a class - skip if it is (we'll get it with the class)
            lines_before = content[:match.start()].split('\n')
            if lines_before and lines_before[-1].strip().startswith('class '):
                continue  
            # Get indentation level
            indent_match = re.match(r'^(\s*)', func_def)
            if indent_match and not indent_match.group(1):  # Only take top-level functions
                snippets.append(func_def)
        
        # Extract classes with their methods
        class_pattern = re.compile(r'class\s+\w+.*?:(.*?)(?=\n\S|\Z)', re.DOTALL)
        for match in class_pattern.finditer(content):
            snippets.append(match.group(0))
        
        # If file has no classes or functions but has content, include the whole file
        if not snippets and len(content.strip()) > 0:
            snippets.append(content)
            
        return snippets
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Build a code corpus for the search model")
    parser.add_argument("--source-dir", required=True, 
                        help="Directory containing Python files to include in corpus")
    parser.add_argument("--output", default="./code_corpus.txt", 
                        help="Output file for the code corpus")
    parser.add_argument("--recursive", action="store_true", 
                        help="Recursively process subdirectories")
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' not found.")
        return
    
    # Find all Python files
    pattern = os.path.join(args.source_dir, "**/*.py") if args.recursive else os.path.join(args.source_dir, "*.py")
    python_files = glob.glob(pattern, recursive=args.recursive)
    
    print(f"Found {len(python_files)} Python files")
    
    # Extract code snippets
    all_snippets = []
    for file_path in python_files:
        file_snippets = extract_functions_and_classes(file_path)
        all_snippets.extend(file_snippets)
        print(f"Extracted {len(file_snippets)} snippets from {os.path.basename(file_path)}")
    
    print(f"Total snippets extracted: {len(all_snippets)}")
    
    # Write to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        for snippet in all_snippets:
            f.write(snippet.strip() + "\n\n---CODE_SEPARATOR---\n\n")
    
    print(f"Code corpus saved to {args.output}")

if __name__ == "__main__":
    main()