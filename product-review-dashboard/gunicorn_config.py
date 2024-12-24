# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 1  
worker_class = 'gthread'
worker_connections = 1000
timeout = 120
preload_app = True
max_requests = 100
max_requests_jitter = 50