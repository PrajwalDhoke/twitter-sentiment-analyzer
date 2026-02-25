"""
Setup Script - Organize Your Project
=====================================
This script helps you organize all files in the correct structure.
"""

import os
import shutil
from pathlib import Path


def setup_project():
    """
    Set up the project structure.
    """
    print("=" * 70)
    print("🔧 PROJECT SETUP SCRIPT")
    print("=" * 70)
    print("\nThis will organize your project files into the correct structure.")
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"\n📂 Working in: {current_dir}")
    
    # Create necessary folders
    print("\n📁 Creating folders...")
    folders = {
        "model": "For trained model files",
        "data": "For dataset files",
        "app": "For API code",
        "frontend": "For UI code"
    }
    
    for folder, description in folders.items():
        folder_path = current_dir / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {folder}/ - {description}")
        else:
            print(f"✓  Exists: {folder}/ - {description}")
    
    # Show expected structure
    print("\n" + "=" * 70)
    print("📋 EXPECTED PROJECT STRUCTURE")
    print("=" * 70)
    print("""
twitter-sentiment-analyzer/
│
├── model/                      ← Trained model files
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/                       ← Dataset
│   └── tweets.csv
│
├── app/                        ← API code
│   ├── main.py
│   └── schemas.py
│
├── frontend/                   ← UI code
│   └── app.py
│
├── train_model.py              ← Training script
├── test_model.py               ← Testing script
├── run_api.py                  ← Run API
├── run_frontend.py             ← Run frontend
├── run_full_app.py             ← Run both
├── requirements.txt            ← Dependencies
├── Dockerfile                  ← Docker config
├── docker-compose.yml          ← Docker compose
└── docker-entrypoint.sh        ← Docker entrypoint
    """)
    
    # Check what's in current directory
    print("=" * 70)
    print("📂 FILES IN CURRENT DIRECTORY")
    print("=" * 70)
    
    all_files = list(current_dir.iterdir())
    py_files = [f for f in all_files if f.suffix == '.py']
    other_files = [f for f in all_files if f.is_file() and f.suffix != '.py']
    folders_found = [f for f in all_files if f.is_dir() and not f.name.startswith('.')]
    
    print(f"\n📄 Python files: {len(py_files)}")
    for f in sorted(py_files):
        print(f"   - {f.name}")
    
    print(f"\n📁 Folders: {len(folders_found)}")
    for f in sorted(folders_found):
        if f.name != '__pycache__':
            print(f"   - {f.name}/")
    
    print(f"\n📋 Other files: {len(other_files)}")
    for f in sorted(other_files):
        if not f.name.startswith('.'):
            print(f"   - {f.name}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 WHAT TO DO NEXT")
    print("=" * 70)
    
    # Check if model files exist
    model_exists = (current_dir / "model" / "sentiment_model.pkl").exists()
    
    if not model_exists:
        print("\n1️⃣  TRAIN YOUR MODEL FIRST:")
        print("   python train_model.py")
        print("   (This creates model files in model/ folder)")
    else:
        print("\n✅ Model files found!")
    
    print("\n2️⃣  TEST YOUR API:")
    print("   python run_api.py")
    print("   (Then open: http://localhost:8000)")
    
    print("\n3️⃣  TEST YOUR FRONTEND:")
    print("   python run_frontend.py")
    print("   (Then open: http://localhost:8501)")
    
    print("\n4️⃣  OR RUN BOTH AT ONCE:")
    print("   python run_full_app.py")
    
    print("\n5️⃣  BUILD DOCKER IMAGE:")
    print("   python docker_helper.py")
    print("   (Choose option 9 for quick start)")
    
    print("\n" + "=" * 70)


def check_critical_files():
    """
    Check if critical files are present.
    """
    print("\n" + "=" * 70)
    print("🎯 CHECKING CRITICAL FILES")
    print("=" * 70)
    
    current_dir = Path.cwd()
    
    critical_files = {
        "train_model.py": "Training script - creates model files",
        "requirements.txt": "Dependencies list",
        "app/main.py": "API server code",
        "app/schemas.py": "API data models",
        "frontend/app.py": "Frontend UI code",
        "run_api.py": "Script to run API",
        "run_frontend.py": "Script to run frontend",
    }
    
    all_present = True
    
    for file_path, description in critical_files.items():
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path:<25} - {description}")
        else:
            print(f"❌ {file_path:<25} - MISSING! {description}")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some files are missing!")
        print("💡 Make sure you downloaded and extracted ALL files from Claude.")
    else:
        print("\n✅ All critical files are present!")
    
    return all_present


def main():
    """
    Main setup function.
    """
    print("\n🚀 TWITTER SENTIMENT ANALYZER - SETUP\n")
    
    # Setup folders
    setup_project()
    
    # Check critical files
    check_critical_files()
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\n💡 TIP: Run 'python check_structure.py' to verify everything!")
    print("\n")
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
