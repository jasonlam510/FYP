"""
Rate Limiter Utility Module

This module provides a rate limiter class for managing API request rates
and concurrent requests.
"""

import asyncio
import time
from .logger import setup_logging

logger = setup_logging(__name__)

class RateLimiter:
    """
    A rate limiter that manages both request rate and concurrent requests.
    
    Attributes:
        max_rpm (int): Maximum requests per minute
        max_active_requests (int): Maximum number of concurrent requests
    """
    
    def __init__(self, max_rpm: int, max_active_requests: int = 500):
        """
        Initialize the rate limiter.
        
        Args:
            max_rpm (int): Maximum requests per minute
            max_active_requests (int): Maximum number of concurrent requests
        """
        self.max_rpm = max_rpm
        self.interval = 60 / max_rpm
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.request_count = 0
        self.start_time = time.time()
        self.active_requests = 0
        self.max_active_requests = max_active_requests
        self.request_lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request, respecting rate limits.
        """
        async with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.interval:
                await asyncio.sleep(self.interval - time_since_last_request)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            # Log performance metrics every 100 requests
            if self.request_count % 100 == 0:
                elapsed_time = current_time - self.start_time
                requests_per_second = self.request_count / elapsed_time
                logger.info(f"Performance metrics - Requests: {self.request_count}, "
                          f"Elapsed time: {elapsed_time:.2f}s, "
                          f"Requests/second: {requests_per_second:.2f}, "
                          f"Active requests: {self.active_requests}")

    async def start_request(self):
        """
        Start a new request, respecting concurrent request limits.
        """
        async with self.request_lock:
            while self.active_requests >= self.max_active_requests:
                await asyncio.sleep(0.1)  # Wait if we're at the limit
            self.active_requests += 1

    async def end_request(self):
        """
        End a request, releasing the concurrent request slot.
        """
        async with self.request_lock:
            self.active_requests -= 1 