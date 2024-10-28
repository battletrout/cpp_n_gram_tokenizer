# build_finder.py

import sys
from pathlib import Path
from typing import Optional, Union, List

class BuildFinder:
    """Utility class to locate and manage C++ build artifacts."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the build finder.
        
        Args:
            project_root: Path to project root. If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.build_dir = self.project_root / "build"
        self._module_path: Optional[Path] = None
    
    def find_build(self, extension: str = ".so") -> Path:
        """
        Find the build directory containing the specified module.
        
        Args:
            extension: File extension to search for (e.g., ".so", ".dll")
            
        Returns:
            Path to the build directory containing the module
            
        Raises:
            FileNotFoundError: If no matching module is found
        """
        # Cache the result if we've already found it
        if self._module_path is not None:
            return self._module_path
        
        # Look for module files with the specified extension
        module_files = list(self.build_dir.rglob(f"*{extension}"))
        
        if not module_files:
            raise FileNotFoundError(
                f"Could not find any {extension} files in {self.build_dir}. "
                "Please build the project first."
            )
        
        # Store the found module path
        self._module_path = module_files[0]
        
        return self._module_path
    
    def add_to_path(self, module_path: Optional[Path] = None) -> None:
        """
        Add the module directory to Python's sys.path.
        
        Args:
            module_path: Optional specific module path to use. If None, finds it automatically.
        """
        if module_path is None:
            module_path = self.find_build()
            
        module_dir = str(module_path.parent)
        if module_dir not in sys.path:
            sys.path.append(module_dir)
            print(f"Added to Python path: {module_dir}")
    
    def get_module_info(self) -> dict:
        """
        Get information about the found module.
        
        Returns:
            Dictionary containing module information
        """
        if self._module_path is None:
            self.find_build()
            
        return {
            "module_path": str(self._module_path),
            "module_name": self._module_path.stem,
            "build_dir": str(self.build_dir),
            "project_root": str(self.project_root)
        }

# Create a default instance for easy importing
default_finder = BuildFinder()

def find_cpp_module():
    """
    Convenience function to find and add the C++ module to Python path.
    Returns the path to the found module.
    """
    module_path = default_finder.find_build()
    default_finder.add_to_path()
    return module_path