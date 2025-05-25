import os
import argparse
import glob
import re

def remove_comments(code):
    
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    
    # Remove inline comments
    lines = []
    for line in code.split('\n'):
     
        in_string = False
        string_char = None
        comment_pos = -1
        
        for i, char in enumerate(line):
            # Track if we're inside a string
            if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            # Find comment character outside of strings
            elif char == '#' and not in_string:
                comment_pos = i
                break
        
        # Add line without comment
        if comment_pos != -1:
            lines.append(line[:comment_pos])
        else:
            lines.append(line)
    
    return '\n'.join(lines)

def extract_functions_and_classes(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments from the entire file first
        content = remove_comments(content)
        
        snippets = []

        function_pattern = re.compile(r'def\s+\w+\s*\(.*?\).*?:(.*?)(?=\n\S|\Z)', re.DOTALL)
        for match in function_pattern.finditer(content):
            func_def = match.group(0)
           
            lines_before = content[:match.start()].split('\n')
            if lines_before and lines_before[-1].strip().startswith('class '):
                continue  
          
            indent_match = re.match(r'^(\s*)', func_def)
            if indent_match and not indent_match.group(1): 
                snippets.append(func_def)

        class_pattern = re.compile(r'class\s+\w+.*?:(.*?)(?=\n\S|\Z)', re.DOTALL)
        for match in class_pattern.finditer(content):
            snippets.append(match.group(0))
        
       
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
    
    
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' not found.")
        return
    
   
    pattern = os.path.join(args.source_dir, "**/*.py") if args.recursive else os.path.join(args.source_dir, "*.py")
    python_files = glob.glob(pattern, recursive=args.recursive)
    
    print(f"Found {len(python_files)} Python files")
    
   
    all_snippets = []
    for file_path in python_files:
        file_snippets = extract_functions_and_classes(file_path)
        all_snippets.extend(file_snippets)
        print(f"Extracted {len(file_snippets)} snippets from {os.path.basename(file_path)}")
    
    print(f"Total snippets extracted: {len(all_snippets)}")
    
  
    with open(args.output, 'w', encoding='utf-8') as f:
        for snippet in all_snippets:
           
            clean_snippet = '\n'.join(line for line in snippet.split('\n') if line.strip())
            f.write(clean_snippet.strip() + "\n\n---CODE_SEPARATOR---\n\n")
    
    print(f"Comment-free code corpus saved to {args.output}")

if __name__ == "__main__":
    main()