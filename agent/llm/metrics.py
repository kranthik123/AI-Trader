from prometheus_client import Counter, Histogram

# A counter to track the number of LLM requests.
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["provider", "model"],
)

# A histogram to track the latency of LLM requests.
LLM_REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "Latency of LLM requests",
    ["provider", "model"],
)

# A counter to track the number of LLM errors.
LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total number of LLM errors",
    ["provider", "model"],
)
