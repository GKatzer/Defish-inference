# api/rate_limiter.py
import time
from collections import defaultdict
from fastapi import HTTPException

from core.config import settings

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        """
        Rate limiter by IP address
        
        Args:
            max_requests: maximum number of requests
            window: time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def check_limit(self, client_ip: str):
        """Checks whether the request limit has been exceeded"""
        now = time.time()
        
        # Remove old requests (older than window seconds)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window
        ]
        
        # Check the limit
        if len(self.requests[client_ip]) >= self.max_requests:
            wait_time = self.window - (now - self.requests[client_ip][0])
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too many requests",
                    "retry_after": int(wait_time),
                    "limit": self.max_requests,
                    "window": self.window
                }
            )
        
        # Add the current request
        self.requests[client_ip].append(now)
        
        return True

# Global limiter instance
# Settings: by default 60 requests per minute from one IP (RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
limiter = RateLimiter(max_requests=settings.rate_limit_requests, window=settings.rate_limit_window)