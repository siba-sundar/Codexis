import os
import torch
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeBERTAnalyzer:
    
    
    def __init__(self):
        
        try:
            logger.info("Initializing CodeBERT analyzer...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Use the CodeBERT model from Microsoft
            model_name = "microsoft/codebert-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            # For code summarization/understanding
            self.nlp = pipeline('feature-extraction', model=model_name, tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)
            
            logger.info("CodeBERT initialization complete")
        except Exception as e:
            logger.error(f"Error initializing CodeBERT: {e}")
            raise
    
    def code_quality_score(self, code):
       
        try:
            # Use the model to get code representation
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            
            # Use the embedding as a feature vector
            embeddings = outputs.logits
            
            # Simple heuristic for code quality based on the embedding vector
            quality_score = torch.sigmoid(torch.mean(embeddings)).item()
            
            return quality_score
        except Exception as e:
            logger.error(f"Error calculating code quality score: {e}")
            return 0.5  # Return neutral score on error
    
    def suggest_improvements(self, code, file_ext):
       
        suggestions = []
        
        # Skip if the code is too long
        if len(code) > 10000:
            suggestions.append("Code is too large for detailed analysis. Consider breaking it into smaller modules.")
            return suggestions
        
        # Skip if the code is empty
        if not code.strip():
            suggestions.append("Code file is empty or contains only whitespace.")
            return suggestions
        
        try:
            # Basic code analysis based on patterns
            
            # Check for long functions/methods
            lines = code.split('\n')
            function_start_patterns = {
                '.py': ['def ', 'async def '],
                '.js': ['function ', '=>', 'async function'],
                '.jsx': ['function ', '=>', 'async function'],
                '.ts': ['function ', '=>', 'async function'],
                '.tsx': ['function ', '=>', 'async function']
            }
            
            current_function_lines = 0
            in_function = False
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts a function definition
                for pattern in function_start_patterns.get(file_ext, []):
                    if pattern in line:
                        in_function = True
                        current_function_lines = 0
                
                # Count lines in the function
                if in_function:
                    current_function_lines += 1
                
                # Check if line ends a function definition
                if in_function and ('}' in line or line.startswith('return')):
                    if current_function_lines > 50:
                        suggestions.append(f"Consider breaking down large function with {current_function_lines} lines into smaller, more focused functions.")
                    in_function = False
            
            # Check for hardcoded values
            if file_ext in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                string_literal_count = len([line for line in lines if '"' in line or "'" in line])
                if string_literal_count > 10:
                    suggestions.append("Consider extracting hardcoded string literals into constants or configuration files.")
            
            # Check for deep nesting
            indentation_levels = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    leading_spaces = len(line) - len(line.lstrip())
                    indentation_levels.append(leading_spaces)
            
            if indentation_levels and max(indentation_levels) > 24:  # Assuming 4 spaces per level, this is 6 levels
                suggestions.append("Deep nesting detected. Consider refactoring to reduce complexity and improve readability.")
            
            # Check for code duplication (simple approach)
            cleaned_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('//')]
            seen_patterns = {}
            for i in range(len(cleaned_lines) - 3):
                pattern = '\n'.join(cleaned_lines[i:i+3])
                if pattern in seen_patterns:
                    seen_patterns[pattern] += 1
                else:
                    seen_patterns[pattern] = 1
            
            duplicate_patterns = [pattern for pattern, count in seen_patterns.items() if count > 1]
            if duplicate_patterns:
                suggestions.append(f"Detected {len(duplicate_patterns)} potentially duplicated code patterns. Consider refactoring into reusable functions.")
                
            # Quality score-based suggestions
            quality_score = self.code_quality_score(code)
            if quality_score < 0.4:
                suggestions.append("Overall code quality appears to be low. Consider refactoring using clean code principles.")
            elif quality_score < 0.6:
                suggestions.append("Code quality is average. Consider adding more documentation and improving naming conventions.")
            
            # Language-specific suggestions
            if file_ext == '.py':
                if 'import *' in code:
                    suggestions.append("Avoid using 'import *' as it can lead to namespace pollution. Import only what you need.")
                if 'except:' in code and 'except Exception:' not in code:
                    suggestions.append("Avoid bare 'except:' clauses. Specify the exceptions you want to catch.")
                if 'global ' in code:
                    suggestions.append("Consider avoiding global variables as they can lead to maintainability issues.")
            
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                if 'var ' in code:
                    suggestions.append("Consider using 'const' or 'let' instead of 'var' for better scoping behavior.")
                if '==' in code and '===' not in code:
                    suggestions.append("Consider using '===' instead of '==' for strict equality comparisons.")
                if 'setTimeout(' in code and 'clearTimeout(' not in code:
                    suggestions.append("Make sure to clear timeouts to prevent memory leaks, especially in React components.")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            suggestions.append("Error analyzing code. Try again with a smaller code snippet.")
            return suggestions
    
    def analyze_repository(self, repo_path, graph):
        
        results = {
            "file_analysis": {},
            "most_complex_files": [],
            "overall_suggestions": []
        }
        
        file_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React JSX',
            '.ts': 'TypeScript',
            '.tsx': 'React TypeScript'
        }
        
        # Get all files that need analysis
        files_to_analyze = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type')
            if node_type in ['python_file', 'js_file']:
                file_path = os.path.join(repo_path, node)
                if os.path.exists(file_path) and os.path.getsize(file_path) < 1000000:  # Skip files larger than 1MB
                    _, ext = os.path.splitext(file_path)
                    if ext in file_extensions:
                        files_to_analyze.append((file_path, ext))
        
        logger.info(f"Analyzing {len(files_to_analyze)} files with CodeBERT...")
        
        # Track file complexity scores
        complexity_scores = {}
        
        # Analyze each file
        for file_path, ext in tqdm(files_to_analyze, desc="Analyzing files"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    code = file.read()
                
                # Skip empty files
                if not code.strip():
                    continue
                
                # Calculate code quality score
                quality_score = self.code_quality_score(code)
                
                # Get improvement suggestions
                suggestions = self.suggest_improvements(code, ext)
                
                # Store the complexity score
                complexity_scores[file_path] = 1.0 - quality_score  # Invert the quality score to get complexity
                
                # Calculate code metrics
                loc = len(code.split('\n'))
                comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#') or line.strip().startswith('//')])
                
                # Store the analysis results
                rel_path = os.path.relpath(file_path, repo_path)
                results["file_analysis"][rel_path] = {
                    "quality_score": quality_score,
                    "language": file_extensions[ext],
                    "lines_of_code": loc,
                    "comment_lines": comment_lines,
                    "comment_ratio": comment_lines / loc if loc > 0 else 0,
                    "suggestions": suggestions
                }
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
        
        # Get the most complex files
        most_complex = sorted(complexity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        results["most_complex_files"] = [os.path.relpath(file_path, repo_path) for file_path, _ in most_complex]
        
        # Generate overall suggestions
        if results["file_analysis"]:
            avg_quality = sum(file["quality_score"] for file in results["file_analysis"].values()) / len(results["file_analysis"])
            avg_comment_ratio = sum(file["comment_ratio"] for file in results["file_analysis"].values()) / len(results["file_analysis"])
            
            if avg_quality < 0.5:
                results["overall_suggestions"].append("The overall code quality appears to be below average. Consider implementing a code review process.")
            
            if avg_comment_ratio < 0.1:
                results["overall_suggestions"].append("The codebase has a low comment ratio. Consider improving documentation to enhance maintainability.")
            
            # Check for consistent code style
            quality_variance = sum((file["quality_score"] - avg_quality) ** 2 for file in results["file_analysis"].values())
            if quality_variance > 0.05:
                results["overall_suggestions"].append("Code quality varies significantly across files. Consider implementing coding standards and linters.")
            
        return results