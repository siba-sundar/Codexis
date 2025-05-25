import os
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyVisualizer:
    
    
    def __init__(self, graph, repo_name):
        
        self.graph = graph
        self.repo_name = repo_name
    
    def create_network_visualization(self, output_path="dependency_graph.html"):
        
        try:
            logger.info("Creating interactive dependency graph visualization...")
            
            # Create a Pyvis network
            net = Network(height="750px", width="100%", directed=True, notebook=False)
            
            # Configure physics and interaction
            net.toggle_physics(True)
            net.set_options("""
            var options = {
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 95,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.1
                    },
                    "minVelocity": 0.75
                },
                "layout": {
                    "improvedLayout": false
                },
                "nodes": {
                    "font": {
                        "size": 12,
                        "face": "Tahoma"
                    }
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "smooth": {
                        "enabled": false
                    }
                },
                "interaction": {
                    "multiselect": true,
                    "navigationButtons": true
                }
            }
            """)
            
            
            node_colors = {
                "python_file": "#3572A5",      
                "js_file": "#F7DF1E",          
                "python_package": "#FFD43B",  
                "js_package": "#F0DB4F"       
            }
            
            
            for node, attr in self.graph.nodes(data=True):
                size = 15
                if attr.get('type') in ['python_package', 'js_package']:
                    size = 10
                
                
                if self.graph.degree(node) > 5:
                    size += 5
                
                net.add_node(
                    node, 
                    label=attr.get('label', node), 
                    title=attr.get('title', node),
                    color=node_colors.get(attr.get('type'), "#CCCCCC"), 
                    size=size
                )
            
           
            for source, target, attr in self.graph.edges(data=True):
                color = "#cccccc"
                if attr.get('type') == 'internal_import':
                    color = "#007bff"  # Blue for internal imports
                elif attr.get('type') == 'external_import':
                    color = "#28a745"  # Green for external imports
                
                net.add_edge(source, target, color=color, arrows='to')
            
            # Add the graph metadata
            metadata = {
                "Repository": self.repo_name,
                "Python Files": self.graph.graph.get('python_files', 0),
                "JavaScript Files": self.graph.graph.get('js_files', 0),
                "Python Packages": self.graph.graph.get('python_packages', 0),
                "JavaScript Packages": self.graph.graph.get('js_packages', 0),
                "Total Nodes": self.graph.number_of_nodes(),
                "Total Edges": self.graph.number_of_edges()
            }
            
          
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Dependency Graph: {self.repo_name}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                    }}
                    .container-fluid {{
                        padding: 20px;
                    }}
                    .header {{
                        background-color: #f8f9fa;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-bottom: 1px solid #e9ecef;
                    }}
                    .graph-container {{
                        height: 750px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        overflow: hidden;
                    }}
                    .metadata {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 4px;
                        margin-bottom: 20px;
                    }}
                    .legend {{
                        margin-top: 20px;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                    }}
                    .legend-item {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 5px;
                    }}
                    .legend-color {{
                        width: 15px;
                        height: 15px;
                        margin-right: 10px;
                        border-radius: 3px;
                    }}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    <div class="header">
                        <h1>Dependency Graph: {self.repo_name}</h1>
                        <p>Interactive visualization of code dependencies</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-3">
                            <div class="metadata">
                                <h4>Repository Information</h4>
                                <ul class="list-group">
            """
            
            
            for key, value in metadata.items():
                html_template += f'<li class="list-group-item d-flex justify-content-between align-items-center">{key} <span class="badge bg-primary rounded-pill">{value}</span></li>\n'
            
          
            html_template += """
                                </ul>
                            </div>
                            
                            <div class="legend">
                                <h4>Legend</h4>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #3572A5;"></div>
                                    <div>Python File</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #F7DF1E;"></div>
                                    <div>JavaScript File</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #FFD43B;"></div>
                                    <div>Python Package</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #F0DB4F;"></div>
                                    <div>JavaScript Package</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #007bff;"></div>
                                    <div>Internal Import</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #28a745;"></div>
                                    <div>External Import</div>
                                </div>
                            </div>
                            
                            <div class="mt-4">
                                <h4>Tips</h4>
                                <ul>
                                    <li>Zoom in/out using mouse wheel</li>
                                    <li>Click and drag to move around</li>
                                    <li>Click on nodes to see details</li>
                                    <li>Double-click to focus on a node</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="col-md-9">
                            <div class="graph-container" id="mynetwork"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Import necessary scripts -->
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            """
            
           
            net.save_graph(output_path)
            
            
            with open(output_path, 'r', encoding='utf-8') as f:
                pyvis_html = f.read()
            
         
            script_start = pyvis_html.find('<script type="text/javascript">')
            script_end = pyvis_html.find('</script>', script_start) + len('</script>')
            vis_script = pyvis_html[script_start:script_end]
            
           
            vis_script_fixed = """
            <script type="text/javascript">
              // initialize global variables.
              var edges;
              var nodes;
              var allNodes;
              var allEdges;
              var nodeColors;
              var originalNodes;
              var network;
              var container;
              var options, data;
              var filter = {
                  item : '',
                  property : '',
                  value : []
              };

              // This method is responsible for drawing the graph, returns the drawn network
              function drawGraph() {
                  // Check if the network container exists before proceeding
                  var container = document.getElementById('mynetwork');
                  if (!container) {
                      console.error("Network container element not found!");
                      return;
                  }
                  
                  try {
                      // Get a DOM element from the document
                      container = document.getElementById('mynetwork');
    
                      // Load the data into vis DataSet
                      nodes = new vis.DataSet(DATA_NODES_PLACEHOLDER);
                      edges = new vis.DataSet(DATA_EDGES_PLACEHOLDER);
                      
                      nodeColors = {};
                      allNodes = nodes.get({ returnType: "Object" });
                      for (nodeId in allNodes) {
                        nodeColors[nodeId] = allNodes[nodeId].color;
                      }
                      allEdges = edges.get({ returnType: "Object" });
                      
                      // adding nodes and edges to the graph
                      data = {nodes: nodes, edges: edges};
    
                      var options = {
                          "physics": {
                              "barnesHut": {
                                  "gravitationalConstant": -2000,
                                  "centralGravity": 0.3,
                                  "springLength": 95,
                                  "springConstant": 0.04,
                                  "damping": 0.09,
                                  "avoidOverlap": 0.1
                              },
                              "minVelocity": 0.75
                          },
                          "layout": {
                              "improvedLayout": false
                          },
                          "nodes": {
                              "font": {
                                  "size": 12,
                                  "face": "Tahoma"
                              }
                          },
                          "edges": {
                              "color": {
                                  "inherit": true
                              },
                              "smooth": {
                                  "enabled": false
                              }
                          },
                          "interaction": {
                              "multiselect": true,
                              "navigationButtons": true
                          }
                      };
    
                      // Initialize the network
                      network = new vis.Network(container, data, options);
                      
                      return network;
                  } catch (error) {
                      console.error("Error in graph rendering:", error);
                      if (container) {
                          container.innerHTML = '<div class="alert alert-danger">Error rendering graph: ' + error.message + '</div>';
                      }
                      return null;
                  }
              }
            </script>
            """
            
           
            nodes_json = json.dumps([
                {
                    'id': node,
                    'label': self.graph.nodes[node].get('label', node),
                    'title': self.graph.nodes[node].get('title', node),
                    'color': node_colors.get(self.graph.nodes[node].get('type'), "#CCCCCC"),
                    'shape': 'dot',
                    'size': 10 if self.graph.nodes[node].get('type') in ['python_package', 'js_package'] else 15
                }
                for node in self.graph.nodes()
            ])
            
            edges_json = json.dumps([
                {
                    'from': source,
                    'to': target,
                    'color': {
                        'color': "#007bff" if self.graph.edges[source, target].get('type') == 'internal_import'
                              else "#28a745" if self.graph.edges[source, target].get('type') == 'external_import'
                              else "#cccccc"
                    },
                    'arrows': 'to'
                }
                for source, target in self.graph.edges()
            ])
            
            
            vis_script_fixed = vis_script_fixed.replace('DATA_NODES_PLACEHOLDER', nodes_json)
            vis_script_fixed = vis_script_fixed.replace('DATA_EDGES_PLACEHOLDER', edges_json)
            
         
            complete_html = html_template + vis_script_fixed + """
            <script>
                // Wait until DOM is fully loaded before initializing the network
                document.addEventListener('DOMContentLoaded', function() {
                    // Safe initialization of the graph
                    try {
                        drawGraph();
                    } catch (error) {
                        console.error("Error initializing graph:", error);
                        var container = document.getElementById('mynetwork');
                        if (container) {
                            container.innerHTML = '<div class="alert alert-danger">Error loading graph. Please check console for details.</div>';
                        }
                    }
                });
            </script>
            </body></html>"""
            
           
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(complete_html)
            
            logger.info(f"Interactive visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
          
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Dependency Graph Error</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-5">
                    <div class="alert alert-danger">
                        <h4>Error Creating Dependency Graph</h4>
                        <p>{str(e)}</p>
                        <pre>{logging.traceback.format_exc()}</pre>
                    </div>
                </div>
            </body>
            </html>
            """
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(error_html)
            raise
    
    # Keep the rest of the methods unchanged
    def create_static_graph(self, output_path="dependency_graph.png"):
       
        try:
            logger.info("Creating static dependency graph visualization...")
            
            # Create a copy of the graph for visualization
            viz_graph = self.graph.copy()
            
          
            node_colors = []
            for node in viz_graph.nodes():
                node_type = viz_graph.nodes[node].get('type')
                if node_type == 'python_file':
                    node_colors.append('#3572A5')  
                elif node_type == 'js_file':
                    node_colors.append('#F7DF1E') 
                elif node_type == 'python_package':
                    node_colors.append('#FFD43B')  
                elif node_type == 'js_package':
                    node_colors.append('#F0DB4F')  
                else:
                    node_colors.append('#CCCCCC')  
            
        
            plt.figure(figsize=(14, 10))
            plt.title(f"Dependency Graph: {self.repo_name}", fontsize=16)
            
            
            if viz_graph.number_of_nodes() > 0:
                try:
                    pos = nx.spring_layout(viz_graph, k=0.15, iterations=50)
                    
                    
                    nx.draw_networkx_nodes(viz_graph, pos, node_size=300, node_color=node_colors, alpha=0.8)
                    nx.draw_networkx_edges(viz_graph, pos, edge_color='#CCCCCC', arrows=True, alpha=0.5)
                    
                   
                    labels = {}
                    for node in viz_graph.nodes():
                        if viz_graph.degree(node) > 3 or viz_graph.nodes[node].get('type') in ['python_package', 'js_package']:
                            labels[node] = viz_graph.nodes[node].get('label', node)
                    
                    nx.draw_networkx_labels(viz_graph, pos, labels=labels, font_size=8)
                    
                  
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3572A5', markersize=10, label='Python File'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F7DF1E', markersize=10, label='JavaScript File'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD43B', markersize=10, label='Python Package'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F0DB4F', markersize=10, label='JS Package')
                    ]
                    plt.legend(handles=legend_elements, loc='upper right')
                    
                   
                    plt.figtext(0.02, 0.02, f"Repository: {self.repo_name}\n"
                               f"Files: Python: {self.graph.graph.get('python_files', 0)}, "
                               f"JavaScript: {self.graph.graph.get('js_files', 0)}\n"
                               f"Packages: Python: {self.graph.graph.get('python_packages', 0)}, "
                               f"JavaScript: {self.graph.graph.get('js_packages', 0)}", 
                               fontsize=10)
                except Exception as layout_error:
                    logger.warning(f"Error in graph layout: {layout_error}. Using random layout instead.")
                    
                    pos = nx.random_layout(viz_graph)
                    nx.draw_networkx(viz_graph, pos, node_color=node_colors, 
                                    edge_color='#CCCCCC', arrows=True, 
                                    node_size=300, font_size=8, alpha=0.8)
                    plt.text(0.5, 0.95, "Warning: Using random layout due to layout calculation error", 
                            ha='center', va='center', fontsize=10, color='red',
                            transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, "No dependencies found", ha='center', va='center', fontsize=14)
            
            
            plt.axis('off')
            
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Static visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating static visualization: {e}")
           
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                plt.savefig(output_path, dpi=100)
                plt.close()
                logger.info(f"Error image saved to {output_path}")
            except Exception as error_img_error:
                logger.error(f"Could not create error image: {error_img_error}")
            raise
    
    def export_graph_data(self, output_path="dependency_data.json"):
        
        try:
            logger.info("Exporting graph data to JSON...")
            
           
            data = {
                "repo_name": self.repo_name,
                "metadata": dict(self.graph.graph),
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for node, attrs in self.graph.nodes(data=True):
                node_data = {"id": node}
                # Ensure all attributes are JSON serializable
                serializable_attrs = {}
                for key, value in attrs.items():
                    try:
                      
                        json.dumps({key: value})
                        serializable_attrs[key] = value
                    except (TypeError, OverflowError):
                      
                        serializable_attrs[key] = str(value)
                
                node_data.update(serializable_attrs)
                data["nodes"].append(node_data)
            
            
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {"source": source, "target": target}
                
                serializable_attrs = {}
                for key, value in attrs.items():
                    try:
                       
                        json.dumps({key: value})
                        serializable_attrs[key] = value
                    except (TypeError, OverflowError):
                        
                        serializable_attrs[key] = str(value)
                
                edge_data.update(serializable_attrs)
                data["edges"].append(edge_data)
            
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Graph data exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            raise

    def safe_remove_directory(self, directory):
       
        try:
            import shutil
            logger.info(f"Attempting to remove directory: {directory}")
            shutil.rmtree(directory)
            logger.info(f"Successfully removed directory: {directory}")
        except PermissionError as e:
            logger.warning(f"Permission error when removing {directory}: {e}")
            try:
              
                import os
                import platform
                import time
                
                
                time.sleep(1)
                
                if platform.system() == 'Windows':
                    logger.info("Using Windows command to force directory removal")
                    os.system(f'rd /s /q "{directory}"')
                else:
                    logger.info("Using Unix command to force directory removal")
                    os.system(f'rm -rf "{directory}"')
                
                
                if not os.path.exists(directory):
                    logger.info(f"Directory successfully removed using system command: {directory}")
                else:
                    logger.warning(f"Failed to remove directory: {directory}")
            except Exception as inner_e:
                logger.error(f"Failed to remove directory {directory} using alternative method: {inner_e}")
        except Exception as e:
            logger.error(f"Error removing directory {directory}: {e}")