import requests
import sys
import json

def test_health():
    response = requests.get('http://localhost:5002/health')
    print(f"Health check status: {response.status_code}")
    print(response.json())

def test_analyze_url(video_url):
    headers = {'Content-Type': 'application/json'}
    data = {'video_url': video_url}
    
    response = requests.post(
        'http://localhost:5002/analyze', 
        headers=headers,
        data=json.dumps(data)
    )
    
    print(f"Analysis status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_health()
    
    # URL de ejemplo para pruebas (reemplaza con una URL válida de video)
    default_video_url = "https://example.com/test-video.mp4"
    
    video_url = sys.argv[1] if len(sys.argv) > 1 else default_video_url
    print(f"\nProbando análisis con video: {video_url}")
    test_analyze_url(video_url)
