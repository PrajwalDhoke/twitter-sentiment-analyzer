"""
Docker Helper Scripts
=====================
Convenient commands for Docker operations.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """
    Run a shell command and handle errors.
    """
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"Running: {command}\n")

    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ Success!")
    return True


def build_image():
    """
    Build Docker image.
    """
    return run_command(
        "docker build -t sentiment-analyzer:latest .",
        "Building Docker Image"
    )


def run_container():
    """
    Run Docker container.
    """
    return run_command(
        "docker run -d -p 8000:8000 -p 8501:8501 --name sentiment-app sentiment-analyzer:latest",
        "Starting Docker Container"
    )


def stop_container():
    """
    Stop Docker container.
    """
    return run_command(
        "docker stop sentiment-app",
        "Stopping Docker Container"
    )


def remove_container():
    """
    Remove Docker container.
    """
    return run_command(
        "docker rm sentiment-app",
        "Removing Docker Container"
    )


def view_logs():
    """
    View container logs.
    """
    return run_command(
        "docker logs sentiment-app",
        "Viewing Container Logs"
    )


def exec_shell():
    """
    Open shell in container.
    """
    return run_command(
        "docker exec -it sentiment-app /bin/bash",
        "Opening Shell in Container"
    )


def compose_up():
    """
    Start with docker-compose.
    """
    return run_command(
        "docker-compose up -d",
        "Starting with Docker Compose"
    )


def compose_down():
    """
    Stop docker-compose.
    """
    return run_command(
        "docker-compose down",
        "Stopping Docker Compose"
    )


def show_menu():
    """
    Show interactive menu.
    """
    print("\n" + "="*70)
    print("🐳 DOCKER HELPER - Twitter Sentiment Analyzer")
    print("="*70)
    print("\nWhat would you like to do?\n")
    print("1. Build Docker Image")
    print("2. Run Container")
    print("3. Stop Container")
    print("4. Remove Container")
    print("5. View Container Logs")
    print("6. Open Shell in Container")
    print("7. Docker Compose Up")
    print("8. Docker Compose Down")
    print("9. Build & Run (Quick Start)")
    print("0. Exit")
    print("\n" + "="*70)


def quick_start():
    """
    Build and run in one go.
    """
    print("\n🚀 QUICK START - Building and Running...")
    
    if not build_image():
        return False
    
    print("\n⏳ Waiting 2 seconds before starting container...")
    import time
    time.sleep(2)
    
    if not run_container():
        return False
    
    print("\n" + "="*70)
    print("✅ APPLICATION IS RUNNING!")
    print("="*70)
    print("\n📍 Access your application:")
    print("   - Frontend: http://localhost:8501")
    print("   - API:      http://localhost:8000")
    print("   - API Docs: http://localhost:8000/")
    print("\n💡 View logs: docker logs sentiment-app")
    print("💡 Stop app:  docker stop sentiment-app")
    print("\n" + "="*70)
    
    return True


def main():
    """
    Main interactive menu.
    """
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == "1":
                build_image()
            elif choice == "2":
                run_container()
            elif choice == "3":
                stop_container()
            elif choice == "4":
                remove_container()
            elif choice == "5":
                view_logs()
            elif choice == "6":
                exec_shell()
            elif choice == "7":
                compose_up()
            elif choice == "8":
                compose_down()
            elif choice == "9":
                quick_start()
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Check if Docker is installed
    result = subprocess.run("docker --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Docker is not installed or not in PATH!")
        print("💡 Install Docker from: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    main()
