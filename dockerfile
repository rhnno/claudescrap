# -------------------------------
# 1. Build Python from source
# -------------------------------
FROM debian:bookworm-slim AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y \
    wget build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev \
    libreadline-dev libsqlite3-dev liblzma-dev tk-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and build Python 3.13.5
RUN wget https://www.python.org/ftp/python/3.13.5/Python-3.13.5.tgz && \
    tar -xvzf Python-3.13.5.tgz && \
    cd Python-3.13.5 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall

# -------------------------------
# 2. Final runtime image
# -------------------------------
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies (no heavy build tools here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip gnupg xvfb supervisor procps bc iputils-ping \
    fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 \
    libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 \
    libnspr4 libnss3 libx11-xcb1 libxcomposite1 \
    libxdamage1 libxrandr2 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy Python from builder stage
COPY --from=builder /usr/local/bin/python3.13 /usr/local/bin/python3.13
COPY --from=builder /usr/local/bin/pip3.13 /usr/local/bin/pip3.13
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13

# Set python/pip alternatives
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.13 1

# Install Google Chrome (latest stable)
RUN wget -q -O- https://dl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-linux.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements (cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . /app

# Make scripts executable & install helper scripts
RUN chmod +x /app/scripts/*.sh \
    && chmod +x /app/docker-configs/*.sh \
    && cp /app/docker-configs/anti-suspend.sh /usr/local/bin/anti-suspend.sh \
    && cp /app/docker-configs/healthcheck.sh /usr/local/bin/healthcheck.sh

# Ensure persistent dirs exist
RUN mkdir -p /app/logs /app/data/raw_html /app/data/processed /app/models

# Supervisor config
COPY docker-configs/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
