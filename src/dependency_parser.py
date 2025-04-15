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
   
    def __init__(self, repo_path):
        
        self.repo_path = repo_path
        
    def parse_python_imports(self, file_path):
        
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
           
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                          
                            module = node.module.split('.')[0]
                            imports.append(module)
            except SyntaxError:
                
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
                   
                    import_regex = r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)'
                    for line in content.split('\n'):
                        match = re.match(import_regex, line)
                        if match:
                            module = match.group(1).split('.')[0]
                            imports.append(module)
                            
            return list(set(imports)) 
        except Exception as e:
            logger.error(f"Error parsing Python imports from {file_path}: {e}")
            return []
    
    def parse_js_imports(self, file_path):
       
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
           
            try:
                tree = esprima.parseModule(content)
           
                def extract_imports(node):
                    if node.type == 'ImportDeclaration':
                        source = node.source.value
                        imports.append(source)
                    elif node.type == 'CallExpression' and node.callee.name == 'require':
                        if node.arguments and node.arguments[0].type == 'Literal':
                            imports.append(node.arguments[0].value)
                    
                    
                    for key, value in node.items():
                        if isinstance(value, dict) and 'type' in value:
                            extract_imports(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'type' in item:
                                    extract_imports(item)
                
                
                extract_imports(tree.toDict())
                
            except Exception as e:
                logger.warning(f"Esprima parsing failed for {file_path}: {e}")
               
                es6_import_regex = r'import\s+(?:.*?\s+from\s+)?[\'"]([^\'"]*)[\'"]\s*;?'
                imports.extend(re.findall(es6_import_regex, content))
              
                require_regex = r'require\([\'"]([^\'"]*)[\'"]'
                imports.extend(re.findall(require_regex, content))
            
            
            cleaned_imports = []
            for imp in imports:
               
                if imp.startswith('.'):
                    continue 
                
           
                if '/' in imp and not imp.startswith('@'):
                    imp = imp.split('/')[0]
                elif imp.startswith('@'):
                 
                    parts = imp.split('/')
                    if len(parts) > 1:
                        imp = f"{parts[0]}/{parts[1]}"
                
                cleaned_imports.append(imp)
                
            return list(set(cleaned_imports)) 
            
        except Exception as e:
            logger.error(f"Error parsing JS imports from {file_path}: {e}")
            return []
    
    def parse_package_json(self, file_path):
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            dependencies = {}
            
           
            if 'dependencies' in data:
                dependencies.update(data['dependencies'])
            
            
            if 'devDependencies' in data:
                dependencies.update(data['devDependencies'])
                
            return dependencies
            
        except Exception as e:
            logger.error(f"Error parsing package.json from {file_path}: {e}")
            return {}
    
    def parse_requirements_txt(self, file_path):
        
        dependencies = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                   
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                  
                    if 'git+' in line:
                        parts = line.split('/')
                        if len(parts) > 1:
                            package = parts[-1].split('@')[0].split('.git')[0]
                            dependencies[package] = 'git'
                        continue
                    
                  
                    for spec in ['==', '>=', '<=', '~=', '>', '<', '=']:
                        if spec in line:
                            package, version = line.split(spec, 1)
                            dependencies[package.strip()] = version.strip()
                            break
                    else:
                        
                        dependencies[line] = 'latest'
                        
            return dependencies
            
        except Exception as e:
            logger.error(f"Error parsing requirements.txt from {file_path}: {e}")
            return {}
    
    def extract_all_dependencies(self):
        
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
        
      
        for root, _, files in os.walk(self.repo_path):
           
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
               
                if file.endswith('.py'):
                    imports = self.parse_python_imports(file_path)
                    if imports:
                        
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        dependencies['python']['files'][rel_path] = imports
                
                
                elif file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    imports = self.parse_js_imports(file_path)
                    if imports:
                       
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        dependencies['javascript']['files'][rel_path] = imports
                
                
                elif file == 'requirements.txt':
                    dependencies['python']['requirements'] = self.parse_requirements_txt(file_path)
                
                
                elif file == 'package.json':
                    dependencies['javascript']['package_json'] = self.parse_package_json(file_path)
        
        return dependencies