from http.server import BaseHTTPRequestHandler
import subprocess
import os
import sys
import threading
import time
import urllib.request

STREAMLIT_PORT = 8501
_server_started = False
_lock = threading.Lock()


def _start_streamlit():
    global _server_started
    with _lock:
        if _server_started:
            return
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
        subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", app_path,
                "--server.port", str(STREAMLIT_PORT),
                "--server.headless", "true",
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://localhost:{STREAMLIT_PORT}/_stcore/health")
                break
            except Exception:
                time.sleep(1)
        _server_started = True


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        _start_streamlit()
        try:
            url = f"http://localhost:{STREAMLIT_PORT}{self.path}"
            req = urllib.request.Request(url)
            for key, val in self.headers.items():
                if key.lower() not in ("host", "connection"):
                    req.add_header(key, val)
            resp = urllib.request.urlopen(req, timeout=30)
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(resp.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
            <html><body style="font-family:sans-serif;max-width:600px;margin:60px auto;padding:20px">
            <h2>Farmley Forecast Dashboard</h2>
            <p>The Streamlit server is starting up. Please refresh in a few seconds.</p>
            <p style="color:#888">Error: {str(e)}</p>
            <script>setTimeout(()=>location.reload(), 5000)</script>
            </body></html>
            """.encode())

    def do_POST(self):
        self.do_GET()

    def log_message(self, format, *args):
        pass
