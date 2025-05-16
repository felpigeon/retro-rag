client<llm> GeminiFlash {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GOOGLE_API_KEY
  }
}
