from pydantic import BaseSettings


class AppConfig(BaseSettings):
    app_name: str = "Investment Data Service"
    debug: bool = False


config = AppConfig()
