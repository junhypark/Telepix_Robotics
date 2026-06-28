FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=osmesa
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libosmesa6 \
    libglfw3 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.lock* ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --dev

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --dev

EXPOSE 8000

CMD ["uv", "run", "robot-sort", "run", "--headless", "--objects", "5", "--seed", "42", "--output", "outputs/run"]
