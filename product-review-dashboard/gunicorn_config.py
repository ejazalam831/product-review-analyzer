# gunicorn_config.py
bind = "0.0.0.0:5000"  # Match the port in Dockerfile
workers = 2
timeout = 120