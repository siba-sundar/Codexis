# GitHub Repository Analyzer

A comprehensive tool for analyzing GitHub repositories, extracting dependencies, building dependency graphs, suggesting code improvements using CodeBERT, and visualizing the results.

## Features

- **Repository Cloning**: Automatically clone GitHub repositories
- **Dependency Extraction**: Analyze Python and JavaScript files to identify dependencies
- **Dependency Graph Building**: Create a graph representation of code dependencies
- **Code Quality Analysis**: Use CodeBERT to analyze code quality and suggest improvements
- **Visualizations**: Generate interactive and static visualizations of the dependency graph
- **Comprehensive Report**: Create a detailed report with findings and recommendations

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/repo-analyzer.git
   cd repo-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic usage:

```bash
python main.py https://github.com/username/repository
```

This will:
1. Clone the repository
2. Parse and extract dependencies for Python and JavaScript
3. Build a dependency graph
4. Use CodeBERT to analyze code quality and suggest improvements
5. Generate visualizations
6. Create a comprehensive report

### Command-line options:

```bash
python main.py --help
```

Output:
```
usage: main.py [-h] [--output-dir OUTPUT_DIR] [--tmp-dir TMP_DIR] [--skip-codebert] [--skip-visualization] [--verbose] repo_url

Analyze GitHub repository dependencies and suggest improvements

positional arguments:
  repo_url              URL of the GitHub repository to analyze

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to store the output files
  --tmp-dir TMP_DIR, -t TMP_DIR
                        Temporary directory for cloning the repository
  --skip-codebert       Skip CodeBERT analysis (faster)
  --skip-visualization  Skip visualization generation
  --verbose, -v         Show verbose output
```

### Examples:

Analyze a repository and save results to a custom directory:
```bash
python main.py https://github.com/username/repository -o my_analysis
```

Skip the CodeBERT analysis for faster processing:
```bash
python main.py https://github.com/username/repository --skip-codebert
```

## Output

The tool generates the following output in the output directory (default: `output/`):

- `dependencies.json`: Raw extracted dependencies
- `graph_analysis.json`: Results of the dependency graph analysis
- `codebert_analysis.json`: Code quality analysis results
- `dependency_graph.html`: Interactive visualization of the dependency graph
- `dependency_graph.png`: Static visualization of the dependency graph
- `graph_data.json`: Raw graph data for external processing
- `report.md`: Comprehensive report with all findings and recommendations

## Visualization Example

The interactive HTML visualization provides a complete view of the project's dependencies:

![Dependency Graph Example](https://via.placeholder.com/800x500.png?text=Dependency+Graph+Visualization)

## How It Works

### 1. Repository Cloning
The tool clones the specified GitHub repository to a temporary directory using GitPython.

### 2. Dependency Parsing
For Python files:
- Parses import statements using AST
- Analyzes requirements.txt files

For JavaScript files:
- Parses ES6 import statements and require() calls
- Analyzes package.json for dependencies

### 3. Graph Building
- Creates a directed graph using NetworkX
- Nodes represent files and external packages
- Edges represent import relationships

### 4. CodeBERT Analysis
- Uses Microsoft's CodeBERT model to analyze code quality
- Identifies complex code patterns
- Generates specific improvement suggestions

### 5. Visualization
- Creates an interactive HTML visualization using Pyvis
- Generates a static PNG visualization using Matplotlib
- Exports raw graph data in JSON format

## Requirements

- Python 3.7+
- Git
- Dependencies listed in requirements.txt

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request."# codebase-analyzer" 
"# codebase-analyzer" 
"# codebase-analyzer" 
