from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ENVIRONMENT: str = 'development'
    ALLOW_ORIGINS: str = '*'
    # GROQ_API_KEY: str
    DEEPGRAM_API_KEY: str
    OPENAI_API_KEY: str
    # OPENAI_PROXY: str | None = None

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()