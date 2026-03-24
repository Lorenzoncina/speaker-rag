bind = "0.0.0.0:8006"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300  # allow long uploads
loglevel = "info"
# Add other options as needed, like accesslog/errorlog if desired
