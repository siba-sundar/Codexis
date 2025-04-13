import os
import shutil
from git import Repo
from git.exc import GitCommandError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clone_repository(repo_url, target_dir="temp_repo"):
    """
    Clone a GitHub repository to the target directory.
    
    Args:
        repo_url (str): URL of the GitHub repository
        target_dir (str): Directory to clone the repository into
        
    Returns:
        str: Path to the cloned repository
    """
    try:
        # Clean up existing directory if it exists
        if os.path.exists(target_dir):
            logger.info(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        logger.info(f"Cloning repository from {repo_url} to {target_dir}")
        Repo.clone_from(repo_url, target_dir)
        logger.info("Repository cloned successfully")
        return target_dir
    
    except GitCommandError as e:
        logger.error(f"Error cloning repository: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def list_files_by_extension(repo_dir, extensions):
    """
    List all files with the specified extensions in the repository.
    
    Args:
        repo_dir (str): Path to the repository
        extensions (list): List of file extensions to look for
        
    Returns:
        dict: Dictionary mapping extensions to lists of file paths
    """
    result = {ext: [] for ext in extensions}
    
    for root, _, files in os.walk(repo_dir):
        # Skip hidden directories like .git
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
            
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    result[ext].append(os.path.join(root, file))
    
    return result