import os

# Get the absolute path to your project root (where this file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_data_path(*paths):
    """
    Returns full path to a file/folder inside the project,
    starting from PROJECT_ROOT.
    Example: get_data_path("Raw_Data", "movies_cleaned.csv")
    """
    return os.path.join(PROJECT_ROOT, *paths)
