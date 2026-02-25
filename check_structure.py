"""
File Structure Checker
======================
This script checks if all required files are in the correct locations.
Run this to diagnose any path issues!
"""

import os
from pathlib import Path


def check_file_structure():
    """
    Check if all required files and folders exist.
    """
    print("=" * 70)
    print("🔍 CHECKING FILE STRUCTURE")
    print("=" * 70)
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"\n📂 Current Directory: {current_dir}")
    
    # Define expected structure
    structure = {
        "model/sentiment_model.pkl": "Model file",
        "model/tfidf_vectorizer.pkl": "Vectorizer file",
        "data/tweets.csv": "Dataset file",
        "app/main.py": "API main file",
        "app/schemas.py": "API schemas",
        "frontend/app.py": "Frontend application",
        "train_model.py": "Training script",
        "test_model.py": "Testing script",
        "run_api.py": "API runner",
        "run_frontend.py": "Frontend runner",
        "run_full_app.py": "Full app runner",
        "requirements.txt": "Dependencies",
        "Dockerfile": "Docker configuration",
        "docker-compose.yml": "Docker Compose config",
        "docker-entrypoint.sh": "Docker entrypoint"
    }
    
    print("\n" + "=" * 70)
    print("📋 CHECKING FILES...")
    print("=" * 70)
    
    all_good = True
    missing_files = []
    found_files = []
    
    for file_path, description in structure.items():
        full_path = current_dir / file_path
        exists = full_path.exists()
        
        if exists:
            size = full_path.stat().st_size
            print(f"✅ {file_path:<40} ({size:,} bytes)")
            found_files.append(file_path)
        else:
            print(f"❌ {file_path:<40} MISSING!")
            missing_files.append(file_path)
            all_good = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Found: {len(found_files)} files")
    print(f"❌ Missing: {len(missing_files)} files")
    
    if missing_files:
        print("\n⚠️  MISSING FILES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 TIP: Make sure you copied all files from the downloads!")
    
    # Check if model files exist (most important!)
    print("\n" + "=" * 70)
    print("🎯 CRITICAL FILES CHECK")
    print("=" * 70)
    
    model_file = current_dir / "model" / "sentiment_model.pkl"
    vectorizer_file = current_dir / "model" / "tfidf_vectorizer.pkl"
    
    if model_file.exists() and vectorizer_file.exists():
        print("✅ Model files found! API should work.")
    else:
        print("❌ Model files missing! You need to run train_model.py first!")
        print(f"\n💡 Run this command:")
        print(f"   python train_model.py")
    
    # Show directory tree
    print("\n" + "=" * 70)
    print("🌳 DIRECTORY TREE")
    print("=" * 70)
    print_tree(current_dir, max_depth=2)
    
    return all_good


def print_tree(directory, prefix="", max_depth=2, current_depth=0):
    """
    Print directory tree structure.
    """
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(items):
            # Skip hidden files and __pycache__
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            
            is_last = i == len(items) - 1
            current = "└── " if is_last else "├── "
            print(f"{prefix}{current}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
                
    except PermissionError:
        pass


def create_missing_folders():
    """
    Create any missing folders.
    """
    print("\n" + "=" * 70)
    print("📁 CREATING MISSING FOLDERS")
    print("=" * 70)
    
    folders = ["model", "data", "app", "frontend"]
    current_dir = Path.cwd()
    
    for folder in folders:
        folder_path = current_dir / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {folder}")
        else:
            print(f"✓  Exists: {folder}")


def main():
    """
    Main function.
    """
    print("\n" + "=" * 70)
    print("🔧 FILE STRUCTURE DIAGNOSTIC TOOL")
    print("=" * 70)
    print("\nThis tool checks if all files are in the correct locations.")
    print("Run this before starting the API or Docker build!\n")
    
    input("Press Enter to start checking...")
    
    # Create folders if missing
    create_missing_folders()
    
    # Check file structure
    all_good = check_file_structure()
    
    # Final message
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL FILES ARE IN PLACE!")
        print("=" * 70)
        print("\n🚀 You can now:")
        print("   1. Run the API: python run_api.py")
        print("   2. Run the Frontend: python run_frontend.py")
        print("   3. Build Docker: docker build -t sentiment-analyzer .")
    else:
        print("⚠️  SOME FILES ARE MISSING!")
        print("=" * 70)
        print("\n💡 Next steps:")
        print("   1. Make sure you copied ALL files from the downloads")
        print("   2. If model files are missing, run: python train_model.py")
        print("   3. Re-run this script to verify")
    
    print("\n")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
