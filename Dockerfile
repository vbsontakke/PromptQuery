FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Configuration comes from the environment (see .env.example). Run as an unprivileged user.
RUN useradd --create-home bot
USER bot
EXPOSE 8080
CMD ["uvicorn", "promptquery.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080"]
