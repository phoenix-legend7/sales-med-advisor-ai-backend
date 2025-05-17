from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ENVIRONMENT: str = 'development'
    ALLOW_ORIGINS: str = '*'
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    OPENAI_API_KEY: str
    OPENAI_ASSISTANT_ID: str

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()