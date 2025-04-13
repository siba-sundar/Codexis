#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import json
from urllib.parse import urlparse

# Import local modules
from clone import clone_repository, list_files_by_extension
from dependency_parser import DependencyParser
from graph_builder import DependencyGraphBuilder
from codebert_analyzer import CodeBERTAnalyzer
from visualizer import DependencyVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('repo_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

def get_repo_name(repo_url):
    """Extract repository name from URL"""
    parsed_url = urlparse(repo_url)
    path = parsed_url.path.strip('/')
    return path.split('/')[-1].replace('.git', '')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze GitHub repository dependencies and suggest improvements'
    )
    parser.add_argument('repo_url', help='URL of the GitHub repository to analyze')
    parser.add_argument('--output-dir', '-o', default='output', help='Directory to store the output files')
    parser.add_argument('--tmp-dir', '-t', default='temp_repo', help='Temporary directory for cloning the repository')
    parser.add_argument('--skip-codebert', action='store_true', help='Skip CodeBERT analysis (faster)')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization generation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    return parser.parse_args()

def main():
    """Main entry point of the application"""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Clone the repository
        logger.info(f"Step 1: Cloning repository from {args.repo_url}")
        repo_dir = clone_repository(args.repo_url, args.tmp_dir)
        repo_name = get_repo_name(args.repo_url)
        logger.info(f"Repository '{repo_name}' cloned successfully")
        
        # Step 2: Parse dependencies
        logger.info("Step 2: Parsing dependencies")
        parser = DependencyParser(repo_dir)
        dependencies = parser.extract_all_dependencies()
        
        # Save raw dependencies
        with open(os.path.join(args.output_dir, 'dependencies.json'), 'w') as f:
            json.dump(dependencies, f, indent=2)
        
        logger.info(f"Found {len(dependencies['python']['files'])} Python files and "
                   f"{len(dependencies['javascript']['files'])} JavaScript files")
        
        # Step 3: Build dependency graph
        logger.info("Step 3: Building dependency graph")
        graph_builder = DependencyGraphBuilder(dependencies)
        graph = graph_builder.build_graph()
        
        # Analyze graph
        analysis_results = graph_builder.analyze_dependencies()
        with open(os.path.join(args.output_dir, 'graph_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Step 4: CodeBERT analysis
        codebert_results = {}
        if not args.skip_codebert:
            logger.info("Step 4: Analyzing code quality with CodeBERT")
            try:
                analyzer = CodeBERTAnalyzer()
                codebert_results = analyzer.analyze_repository(repo_dir, graph)
                with open(os.path.join(args.output_dir, 'codebert_analysis.json'), 'w') as f:
                    json.dump(codebert_results, f, indent=2)
            except Exception as e:
                logger.error(f"CodeBERT analysis failed: {e}")
                logger.info("Continuing without CodeBERT analysis")
        else:
            logger.info("CodeBERT analysis skipped")
        
        # Step 5: Visualize results
        if not args.skip_visualization:
            logger.info("Step 5: Generating visualizations")
            visualizer = DependencyVisualizer(graph, repo_name)
            
            # Create interactive HTML visualization
            html_path = os.path.join(args.output_dir, 'dependency_graph.html')
            visualizer.create_network_visualization(html_path)
            
            # Create static visualization
            png_path = os.path.join(args.output_dir, 'dependency_graph.png')
            visualizer.create_static_graph(png_path)
            
            # Export graph data
            json_path = os.path.join(args.output_dir, 'graph_data.json')
            visualizer.export_graph_data(json_path)
        else:
            logger.info("Visualization generation skipped")
        
        # Generate final report
        generate_report(args.output_dir, repo_name, dependencies, analysis_results, codebert_results)
        
        # Cleanup
        if os.path.exists(repo_dir) and repo_dir.startswith("temp_"):
            import shutil
            logger.info(f"Cleaning up temporary directory: {repo_dir}")
            shutil.rmtree(repo_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {os.path.abspath(args.output_dir)}")
        
        print("\n=== Repository Analysis Summary ===")
        print(f"Repository: {repo_name}")
        print(f"Python files: {graph.graph.get('python_files', 0)}")
        print(f"JavaScript files: {graph.graph.get('js_files', 0)}")
        print(f"Python packages: {graph.graph.get('python_packages', 0)}")
        print(f"JavaScript packages: {graph.graph.get('js_packages', 0)}")
        print(f"Total nodes in dependency graph: {graph.number_of_nodes()}")
        print(f"Total edges in dependency graph: {graph.number_of_edges()}")
        
        if not args.skip_visualization:
            print(f"\nHTML visualization: {os.path.abspath(html_path)}")
        
        print(f"\nFull report: {os.path.abspath(os.path.join(args.output_dir, 'report.md'))}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    
    return 0

def generate_report(output_dir, repo_name, dependencies, graph_analysis, codebert_results):
    """Generate a Markdown report with the analysis results"""
    report_path = os.path.join(output_dir, 'report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Repository Analysis Report: {repo_name}\n\n")
        f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Repository statistics
        f.write("## Repository Statistics\n\n")
        
        # Python stats
        py_files = len(dependencies['python']['files'])
        py_packages = len(dependencies['python']['requirements'])
        f.write(f"### Python\n")
        f.write(f"- **Files:** {py_files}\n")
        f.write(f"- **External Packages:** {py_packages}\n")
        
        if py_packages > 0:
            f.write("\n**Top Python Dependencies:**\n\n")
            f.write("| Package | Version |\n")
            f.write("|---------|--------|\n")
            
            # Get the top dependencies
            top_deps = list(dependencies['python']['requirements'].items())
            for package, version in sorted(top_deps)[:10]:  # Show top 10
                f.write(f"| {package} | {version} |\n")
        
        # JavaScript stats
        js_files = len(dependencies['javascript']['files'])
        js_packages = len(dependencies['javascript']['package_json'])
        f.write(f"\n### JavaScript\n")
        f.write(f"- **Files:** {js_files}\n")
        f.write(f"- **External Packages:** {js_packages}\n")
        
        if js_packages > 0:
            f.write("\n**Top JavaScript Dependencies:**\n\n")
            f.write("| Package | Version |\n")
            f.write("|---------|--------|\n")
            
            # Get the top dependencies
            top_deps = list(dependencies['javascript']['package_json'].items())
            for package, version in sorted(top_deps)[:10]:  # Show top 10
                f.write(f"| {package} | {version} |\n")
        
        # Graph analysis
        f.write("\n## Dependency Graph Analysis\n\n")
        f.write(f"- **Total Nodes:** {graph_analysis['stats']['total_nodes']}\n")
        f.write(f"- **Total Edges:** {graph_analysis['stats']['total_edges']}\n")
        f.write(f"- **Graph Density:** {graph_analysis['density']:.4f}\n")
        f.write(f"- **Average Clustering:** {graph_analysis['avg_clustering']:.4f}\n")
        f.write(f"- **Number of Clusters:** {graph_analysis['clusters']}\n")
        
        # Central nodes
        if 'central_nodes' in graph_analysis and graph_analysis['central_nodes']:
            f.write("\n### Most Central Components\n\n")
            
            # Degree centrality
            if 'degree' in graph_analysis['central_nodes']:
                f.write("**Most Connected Components (Degree Centrality):**\n\n")
                for node, centrality in graph_analysis['central_nodes']['degree']:
                    f.write(f"- {node} ({centrality:.4f})\n")
            
            # PageRank
            if 'pagerank' in graph_analysis['central_nodes']:
                f.write("\n**Most Important Components (PageRank):**\n\n")
                for node, centrality in graph_analysis['central_nodes']['pagerank']:
                    f.write(f"- {node} ({centrality:.4f})\n")
        
        # Cycles
        if 'cycles' in graph_analysis and graph_analysis['cycles']:
            f.write("\n### Circular Dependencies\n\n")
            f.write("The following circular dependencies were detected:\n\n")
            for cycle in graph_analysis['cycles']:
                f.write(f"- {' → '.join(cycle)} → {cycle[0]}\n")
        
        # Recommendations from graph analysis
        if 'recommendations' in graph_analysis and graph_analysis['recommendations']:
            f.write("\n### Graph Structure Recommendations\n\n")
            for rec in graph_analysis['recommendations']:
                f.write(f"- {rec}\n")
        
        # CodeBERT analysis
        if codebert_results:
            f.write("\n## Code Quality Analysis\n\n")
            
            # Most complex files
            # Most complex files
            if 'most_complex_files' in codebert_results and codebert_results['most_complex_files']:
                f.write("### Most Complex Files\n\n")
                for file in codebert_results['most_complex_files']:
                    f.write(f"- `{file}`\n")
            
            # File quality overview
            if 'file_analysis' in codebert_results and codebert_results['file_analysis']:
                f.write("\n### Code Quality Metrics\n\n")
                f.write("| File | Language | Lines | Comment Ratio | Quality Score |\n")
                f.write("|------|----------|-------|---------------|---------------|\n")
                
                # Get top files sorted by quality score (ascending)
                files_by_quality = sorted(
                    codebert_results['file_analysis'].items(),
                    key=lambda x: x[1]['quality_score']
                )
                
                # Show top 10 worst and top 10 best files
                for file, data in files_by_quality[:10]:  # 10 worst
                    f.write(f"| `{file}` | {data['language']} | {data['lines_of_code']} | {data['comment_ratio']:.2f} | {data['quality_score']:.2f} |\n")
                
                f.write("\n...\n\n")
                
                for file, data in files_by_quality[-10:]:  # 10 best
                    f.write(f"| `{file}` | {data['language']} | {data['lines_of_code']} | {data['comment_ratio']:.2f} | {data['quality_score']:.2f} |\n")
            
            # Overall suggestions
            if 'overall_suggestions' in codebert_results and codebert_results['overall_suggestions']:
                f.write("\n### Overall Code Quality Suggestions\n\n")
                for suggestion in codebert_results['overall_suggestions']:
                    f.write(f"- {suggestion}\n")
            
            # Sample file-specific suggestions
            sample_count = 0
            if 'file_analysis' in codebert_results:
                f.write("\n### Sample File-Specific Suggestions\n\n")
                for file, data in codebert_results['file_analysis'].items():
                    if 'suggestions' in data and data['suggestions'] and sample_count < 5:
                        f.write(f"**File: `{file}`**\n\n")
                        for suggestion in data['suggestions'][:3]:  # Show top 3 suggestions per file
                            f.write(f"- {suggestion}\n")
                        f.write("\n")
                        sample_count += 1
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This report provides an overview of the repository structure, dependencies, and code quality. ")
        f.write("Use the interactive visualization for a more detailed exploration of the dependency relationships.\n\n")
        f.write("To improve the codebase, focus on addressing circular dependencies, refactoring complex files, and following the provided recommendations.")
        
        logger.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    sys.exit(main())