import os
import hashlib
import json
import redis.asyncio as redis
from aiocache import Cache as AIOCache

class Cache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self.redis = redis.from_url(redis_url)
            self.use_redis = True
        else:
            self.cache = AIOCache(AIOCache.MEMORY)
            self.use_redis = False

    async def get(self, key):
        if self.use_redis:
            return await self.redis.get(key)
        else:
            return await self.cache.get(key)

    async def set(self, key, value, ttl=60):
        if self.use_redis:
            await self.redis.set(key, value, ex=ttl)
        else:
            await self.cache.set(key, value, ttl=ttl)

    def get_cache_key(self, provider, model, prompt, **kwargs):
        """Creates a cache key based on a hash of the provider, model, prompt, and options."""
        payload = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "options": kwargs
        }

        # Use json.dumps with sort_keys=True to ensure consistent hashing
        payload_str = json.dumps(payload, sort_keys=True)

        # Use hashlib to create a sha256 hash of the payload
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
