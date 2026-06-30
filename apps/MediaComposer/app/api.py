import os
import sys
import socket
import subprocess
from flask import Blueprint, jsonify

# Setup paths
mediacomposer_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mediacomposer_dir in sys.path:
    sys.path.remove(mediacomposer_dir)
sys.path.insert(0, mediacomposer_dir)

composer_bp = Blueprint('composer', __name__)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

@composer_bp.route('/launch', methods=['GET'])
def launch_composer():
    port = 8502
    url = f"http://127.0.0.1:{port}"
    
    if is_port_in_use(port):
        return jsonify({"success": True, "url": url, "message": "Media Composer already running."})
        
    try:
        # Resolve virtual environment python relative to this file
        # __file__ is AIVoice/apps/MediaComposer/app/api.py
        # MediaComposer is parent of app, apps is parent of MediaComposer, AIVoice is parent of apps
        aivoice_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if sys.platform == "win32":
            venv_python = os.path.join(aivoice_dir, ".venv", "Scripts", "python.exe")
        else:
            venv_python = os.path.join(aivoice_dir, ".venv", "bin", "python")
        
        if not os.path.exists(venv_python):
            # Fallback to standard python in path if .venv python is missing
            venv_python = "python"
            
        mc_dir = os.path.join(aivoice_dir, "apps", "MediaComposer")
        
        # Start streamlit in background without opening cmd window
        cmd = [venv_python, "-m", "streamlit", "run", "webui/Main.py", "--server.port", str(port), "--server.address", "127.0.0.1"]
        
        # Windows-specific process creation flag to hide console
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW
            
        subprocess.Popen(
            cmd,
            cwd=mc_dir,
            creationflags=creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        return jsonify({
            "success": True,
            "url": url,
            "message": "Media Composer has been launched successfully."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to launch Media Composer: {str(e)}"
        }), 500
