import os
import re
import json
import logging
import ast
import astroid
import esprima
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyParser:
    """Class to parse dependencies from Python and JavaScript files"""
    
    def __init__(self, repo_path):
        """
        Initialize the parser with the repository path
        
        Args:
            repo_path (str): Path to the cloned repository
        """
        self.repo_path = repo_path
        
    def parse_python_imports(self, file_path):
        """
        Extract Python import statements from a file
        
        Args:
            file_path (str): Path to the Python file
            
        Returns:
            list: List of imported modules
        """
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try parsing with ast first
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Get the top-level module
                            module = node.module.split('.')[0]
                            imports.append(module)
            except SyntaxError:
                # Fallback to using astroid if ast fails
                try:
                    tree = astroid.parse(content)
                    for node in tree.nodes_of_class((astroid.Import, astroid.ImportFrom)):
                        if isinstance(node, astroid.Import):
                            for name in node.names:
                                imports.append(name[0])
                        elif isinstance(node, astroid.ImportFrom):
                            if node.modname:
                                module = node.modname.split('.')[0]
                                imports.append(module)
                except Exception as e:
                    logger.warning(f"Astroid parsing failed for {file_path}: {e}")
                    # Use regex as last resort
                    import_regex = r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)'
                    for line in content.split('\n'):
                        match = re.match(import_regex, line)
                        if match:
                            module = match.group(1).split('.')[0]
                            imports.append(module)
                            
            return list(set(imports))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error parsing Python imports from {file_path}: {e}")
            return []
    
    def parse_js_imports(self, file_path):
        """
        Extract JavaScript import statements and require() calls from a file
        
        Args:
            file_path (str): Path to the JavaScript file
            
        Returns:
            list: List of imported modules
        """
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try parsing with esprima
            try:
                tree = esprima.parseModule(content)
                
                # Function to recursively extract imports from AST
                def extract_imports(node):
                    if node.type == 'ImportDeclaration':
                        source = node.source.value
                        imports.append(source)
                    elif node.type == 'CallExpression' and node.callee.name == 'require':
                        if node.arguments and node.arguments[0].type == 'Literal':
                            imports.append(node.arguments[0].value)
                    
                    # Recursively check child nodes
                    for key, value in node.items():
                        if isinstance(value, dict) and 'type' in value:
                            extract_imports(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'type' in item:
                                    extract_imports(item)
                
                # Start extraction from the root node
                extract_imports(tree.toDict())
                
            except Exception as e:
                logger.warning(f"Esprima parsing failed for {file_path}: {e}")
                # Use regex as fallback
                # Match ES6 imports
                es6_import_regex = r'import\s+(?:.*?\s+from\s+)?[\'"]([^\'"]*)[\'"]\s*;?'
                imports.extend(re.findall(es6_import_regex, content))
                
                # Match require statements
                require_regex = r'require\([\'"]([^\'"]*)[\'"]'
                imports.extend(re.findall(require_regex, content))
            
            # Process the imports to clean them up
            cleaned_imports = []
            for imp in imports:
                # Remove relative path indicators and get the package name
                if imp.startswith('.'):
                    continue  # Skip relative imports
                
                # For npm packages, get the main package name
                if '/' in imp and not imp.startswith('@'):
                    imp = imp.split('/')[0]
                elif imp.startswith('@'):
                    # Handle scoped packages like @angular/core
                    parts = imp.split('/')
                    if len(parts) > 1:
                        imp = f"{parts[0]}/{parts[1]}"
                
                cleaned_imports.append(imp)
                
            return list(set(cleaned_imports))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error parsing JS imports from {file_path}: {e}")
            return []
    
    def parse_package_json(self, file_path):
        """
        Extract dependencies from package.json file
        
        Args:
            file_path (str): Path to the package.json file
            
        Returns:
            dict: Dictionary of dependencies
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            dependencies = {}
            
            # Extract dependencies
            if 'dependencies' in data:
                dependencies.update(data['dependencies'])
            
            # Extract devDependencies
            if 'devDependencies' in data:
                dependencies.update(data['devDependencies'])
                
            return dependencies
            
        except Exception as e:
            logger.error(f"Error parsing package.json from {file_path}: {e}")
            return {}
    
    def parse_requirements_txt(self, file_path):
        """
        Extract dependencies from requirements.txt file
        
        Args:
            file_path (str): Path to the requirements.txt file
            
        Returns:
            dict: Dictionary of dependencies
        """
        dependencies = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle GitHub URLs
                    if 'git+' in line:
                        parts = line.split('/')
                        if len(parts) > 1:
                            package = parts[-1].split('@')[0].split('.git')[0]
                            dependencies[package] = 'git'
                        continue
                    
                    # Regular requirements format
                    # Split on common version specifiers
                    for spec in ['==', '>=', '<=', '~=', '>', '<', '=']:
                        if spec in line:
                            package, version = line.split(spec, 1)
                            dependencies[package.strip()] = version.strip()
                            break
                    else:
                        # No version specifier found
                        dependencies[line] = 'latest'
                        
            return dependencies
            
        except Exception as e:
            logger.error(f"Error parsing requirements.txt from {file_path}: {e}")
            return {}
    
    def extract_all_dependencies(self):
        """
        Extract all dependencies from the repository
        
        Returns:
            dict: Dictionary with Python and JavaScript dependencies
        """
        dependencies = {
            'python': {
                'files': {},
                'requirements': {}
            },
            'javascript': {
                'files': {},
                'package_json': {}
            }
        }
        
        # Find all Python files
        for root, _, files in os.walk(self.repo_path):
            # Skip hidden directories like .git
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # Process Python files
                if file.endswith('.py'):
                    imports = self.parse_python_imports(file_path)
                    if imports:
                        # Make path relative to repo
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        dependencies['python']['files'][rel_path] = imports
                
                # Process JavaScript files
                elif file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    imports = self.parse_js_imports(file_path)
                    if imports:
                        # Make path relative to repo
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        dependencies['javascript']['files'][rel_path] = imports
                
                # Process requirements.txt
                elif file == 'requirements.txt':
                    dependencies['python']['requirements'] = self.parse_requirements_txt(file_path)
                
                # Process package.json
                elif file == 'package.json':
                    dependencies['javascript']['package_json'] = self.parse_package_json(file_path)
        
        return dependencies