# gunicorn.conf.py — applied automatically when Gunicorn starts
# Render uses: gunicorn app:app (picks this file up automatically)

# Workers & concurrency
workers = 1          # 1 is enough for Render free tier (512 MB RAM)
threads = 2
timeout = 120        # seconds before a worker is killed

# ── Request-size limits ────────────────────────────────────────────────────
# The default limit_request_line (4094 bytes) and limit_request_field_size
# (8190 bytes) only affect HTTP headers — NOT the body.
# The body limit is controlled by Flask's MAX_CONTENT_LENGTH (set in app.py).
# We raise these header limits just in case large cookie / auth headers appear.
limit_request_line        = 8190
limit_request_field_size  = 8190
limit_request_fields      = 100
