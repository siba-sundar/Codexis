import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyGraphBuilder:
    """Build a dependency graph from parsed dependencies"""
    
    def __init__(self, dependencies):
        """
        Initialize with extracted dependencies
        
        Args:
            dependencies (dict): The extracted dependencies from the parser
        """
        self.dependencies = dependencies
        self.graph = nx.DiGraph()
        
    def build_graph(self):
        """
        Build the dependency graph
        
        Returns:
            nx.DiGraph: Constructed NetworkX directed graph
        """
        logger.info("Building dependency graph...")
        
        # Process Python files
        python_files = self.dependencies['python']['files']
        for file_path, imports in python_files.items():
            # Add node for the file
            self.graph.add_node(file_path, 
                               type='python_file', 
                               label=file_path,
                               title=f"Python: {file_path}")
            
            # Add edges to imported modules
            for module in imports:
                # Check if it's an internal module or a package
                if module in python_files:
                    # It's an internal module
                    self.graph.add_edge(file_path, module, type='internal_import')
                else:
                    # It's an external package
                    module_node = f"py_pkg:{module}"
                    if not self.graph.has_node(module_node):
                        version = self.dependencies['python']['requirements'].get(module, 'unknown')
                        self.graph.add_node(module_node, 
                                          type='python_package', 
                                          label=module,
                                          title=f"Python Package: {module} ({version})")
                    
                    self.graph.add_edge(file_path, module_node, type='external_import')
        
        # Process JavaScript files
        js_files = self.dependencies['javascript']['files']
        for file_path, imports in js_files.items():
            # Add node for the file
            self.graph.add_node(file_path, 
                               type='js_file', 
                               label=file_path,
                               title=f"JavaScript: {file_path}")
            
            # Add edges to imported modules
            for module in imports:
                # Check if it's an internal module or a package
                if module in js_files:
                    # It's an internal module
                    self.graph.add_edge(file_path, module, type='internal_import')
                else:
                    # It's an external package
                    module_node = f"js_pkg:{module}"
                    if not self.graph.has_node(module_node):
                        version = self.dependencies['javascript']['package_json'].get(module, 'unknown')
                        self.graph.add_node(module_node, 
                                          type='js_package', 
                                          label=module,
                                          title=f"JS Package: {module} ({version})")
                    
                    self.graph.add_edge(file_path, module_node, type='external_import')
        
        # Add metadata to the graph
        self.graph.graph['python_files'] = len(python_files)
        self.graph.graph['js_files'] = len(js_files)
        self.graph.graph['python_packages'] = len(self.dependencies['python']['requirements'])
        self.graph.graph['js_packages'] = len(self.dependencies['javascript']['package_json'])
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_central_nodes(self, top_n=5):
        """
        Get the most central nodes in the graph based on different centrality measures
        
        Args:
            top_n (int): Number of top nodes to return
            
        Returns:
            dict: Dictionary with different centrality measures
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        centrality = {}
        
        # Degree centrality
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            centrality['degree'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as e:
            logger.warning(f"Error calculating degree centrality: {e}")
        
        # Betweenness centrality
        try:
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            centrality['betweenness'] = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as e:
            logger.warning(f"Error calculating betweenness centrality: {e}")
        
        # PageRank
        try:
            pagerank = nx.pagerank(self.graph)
            centrality['pagerank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as e:
            logger.warning(f"Error calculating PageRank: {e}")
        
        return centrality
    
    def identify_clusters(self):
        """
        Identify clusters in the graph using community detection
        
        Returns:
            dict: Dictionary mapping nodes to their community
        """
        try:
            # Create undirected copy of the graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use Louvain method for community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(undirected_graph)
                return partition
            except ImportError:
                # Fallback to connected components
                logger.warning("Community detection package not found, using connected components")
                components = list(nx.connected_components(undirected_graph))
                partition = {}
                for i, component in enumerate(components):
                    for node in component:
                        partition[node] = i
                return partition
                
        except Exception as e:
            logger.warning(f"Error identifying clusters: {e}")
            return {}
    
    def analyze_dependencies(self):
        """
        Perform analysis on the dependency graph
        
        Returns:
            dict: Analysis results
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "error": "Empty graph, no dependencies to analyze"
            }
        
        analysis = {
            "stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "python_files": self.graph.graph.get('python_files', 0),
                "js_files": self.graph.graph.get('js_files', 0),
                "python_packages": self.graph.graph.get('python_packages', 0),
                "js_packages": self.graph.graph.get('js_packages', 0),
            },
            "central_nodes": self.get_central_nodes(),
            "clusters": len(set(self.identify_clusters().values())) if self.identify_clusters() else 0,
            "density": nx.density(self.graph),
            "avg_clustering": nx.average_clustering(self.graph.to_undirected()),
            "recommendations": []
        }
        
        # Generate recommendations based on the analysis
        if analysis["density"] > 0.7:
            analysis["recommendations"].append("High graph density suggests tight coupling. Consider modularizing the codebase.")
        
        if analysis["avg_clustering"] < 0.2:
            analysis["recommendations"].append("Low clustering coefficient suggests poor code organization. Consider grouping related functionality.")
        
        # Check for dependency cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                analysis["recommendations"].append(f"Found {len(cycles)} circular dependencies. Consider refactoring to remove cycles.")
                analysis["cycles"] = [cycle for cycle in cycles[:5]]  # Show up to 5 cycles
        except Exception as e:
            logger.warning(f"Error detecting cycles: {e}")
        
        # Get orphan nodes (files that don't import anything or aren't imported)
        orphans = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        if orphans:
            analysis["recommendations"].append(f"Found {len(orphans)} orphaned files that aren't connected to the dependency graph.")
            analysis["orphans"] = orphans[:10]  # Show up to 10 orphans
        
        return analysis