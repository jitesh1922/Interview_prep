import time
import threading

class TokenBucket:
    def __init__(self, rate_limit: int, time_window: int, max_credits_multiplier: float = 2.0):
        self.capacity = rate_limit * max_credits_multiplier  # Max tokens = credits
        self.tokens = float(rate_limit)
        self.refill_rate = rate_limit / time_window
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens
            refill_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + refill_tokens)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False


class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.user_buckets = {}
        self.global_lock = threading.Lock()

    def get_bucket_for_user(self, user_id: str) -> TokenBucket:
        with self.global_lock:
            if user_id not in self.user_buckets:
                self.user_buckets[user_id] = TokenBucket(self.rate_limit, self.time_window)
            return self.user_buckets[user_id]

    def is_request_allowed(self, user_id: str) -> bool:
        bucket = self.get_bucket_for_user(user_id)
        return bucket.allow_request()


def test_rate_limiter():
    limiter = RateLimiter(rate_limit=5, time_window=10)  # 5 req per 10 sec
    user = "alice"

    allowed = 0
    for _ in range(5):
        if limiter.is_request_allowed(user):
            allowed += 1

    assert allowed == 5
    assert limiter.is_request_allowed(user) == False  # Should be rate limited

    print("Test passed.")

test_rate_limiter()
