import os
import subprocess
from datetime import datetime

def dump_directory():
    output_file = "directory_structure.dir"
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"Directory Structure and Contents\n")
        f.write(f"Generated on {datetime.now()}\n")
        f.write("-" * 40 + "\n\n")
        
        # Get directory structure using tree command
        try:
            tree_output = subprocess.check_output(['tree', '-a']).decode()
            f.write("DIRECTORY STRUCTURE:\n")
            f.write(tree_output + "\n")
        except:
            f.write("Note: 'tree' command not found. Install it for directory structure.\n\n")
        
        # Get list of files excluding those in .gitignore
        try:
            tracked_files = subprocess.check_output(['git', 'ls-files']).decode().splitlines()
        except:
            # If not a git repo, just get all files
            tracked_files = []
            for root, _, files in os.walk('.'):
                tracked_files.extend(os.path.join(root, file) for file in files)
        
        f.write("FILE CONTENTS:\n")
        for file_path in tracked_files:
            try:
                with open(file_path, 'r') as file:
                    f.write(f"\n=== {file_path} ===\n")
                    f.write(file.read() + "\n")
            except:
                f.write(f"Error reading {file_path}\n")

if __name__ == "__main__":
    dump_directory()