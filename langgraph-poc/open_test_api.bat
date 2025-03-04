@echo off
echo === Starting HTTP Server for test_api.html ===
echo === Press Ctrl+C to stop the server when finished ===
echo.

start "" "http://localhost:8000/test_api.html"
python -m http.server 8000 