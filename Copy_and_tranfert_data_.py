# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:43:04 2023

@author: mzang
"""

# Import necessary modules
import os       # Provides a way to interact with the operating system
import shutil   # Offers high-level file operations

# Define a function to copy files from a source folder to a destination folder
def copyFile(source_folder, new_folder):
    # Check if the destination folder exists, create it if not
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # List all files in the source folder
    source_files = os.listdir(source_folder)

    # Iterate through each file in the source folder
    for file_name in source_files:
        # Build the full path of the source file
        source_path = os.path.join(source_folder, file_name)
        
        # Build the full path of the target file in the destination folder
        target_path = os.path.join(new_folder, file_name)
        
        # Copy the source file to the target folder
        shutil.copy(source_path, target_path)

    # Display a message once file copying is complete
    print("File copying completed.")

