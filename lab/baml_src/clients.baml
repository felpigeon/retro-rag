client<llm> GeminiFlash {
  provider google-ai
  retry_policy RetryPolicy
  options {
    model "gemini-2.0-flash"
  }
}

retry_policy RetryPolicy {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 60000
  }
}
