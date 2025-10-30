import httpx

# Create a shared httpx.AsyncClient instance for connection pooling
http_client = httpx.AsyncClient(timeout=30)

def get_http_client() -> httpx.AsyncClient:
    """Returns the shared httpx.AsyncClient instance."""
    return http_client
