"""
Quick test script to verify the backend is working.
Run this after starting the server to test the /convert endpoint.
"""
import requests
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_server():
    """Test if the server is running and responding."""
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
        print("✅ Server is running!")
        print(f"   Status: {response.status_code}")
        print("   Visit http://127.0.0.1:8000/docs for API documentation")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is NOT running!")
        print("   Please start the server first:")
        print("   1. Activate virtual environment: ..\\sop\\Scripts\\activate.bat")
        print("   2. Run: uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

if __name__ == "__main__":
    test_server()

