# syntax=docker/dockerfile:1
# Enhanced ML-Powered Web Scraper
FROM python:3.11-slim-bookworm

LABEL maintainer="Enhanced Analyzer System" \
      description="ML-powered smart web scraper with configurable pagination detection" \
      version="2.0"

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget=1.21.3-1+b2 \
    curl=7.88.1-10+deb12u5 \
    unzip=6.0-28 \
    gnupg=2.2.40-1.1 \
    xvfb=2:21.1.7-3+deb12u7 \
    supervisor=4.2.1-1+deb12u1 \
    procps=2:4.0.2-3 \
    fonts-liberation=1:1.07.4-11 \
    libasound2=1.2.8-1+b1 \
    libatk-bridge2.0-0=2.46.0-5 \
    libatk1.0-0=2.46.0-5 \
    libcups2=2.4.2-3+deb12u5 \
    libdbus-1-3=1.14.10-1~deb12u1 \
    libdrm2=2.4.114-1+b1 \
    libgbm1=22.3.6-1+deb12u1 \
    libgtk-3-0=3.24.38-2~deb12u1 \
    libnspr4=2:4.35-1 \
    libnss3=2:3.87.1-1 \
    libx11-xcb1=2:1.8.4-2+deb12u2 \
    libxcomposite1=1:0.4.5-1 \
    libxdamage1=1:1.1.6-1 \
    libxrandr2=2:1.5.2-2+b1 \
    xdg-utils=1.1.3-4.1 \
    build-essential=12.9 \
    pkg-config=1.8.1-1 \
    libhdf5-dev=1.10.8+repack1-1 \
    libopenblas-dev=0.3.21+ds-4 \
    liblapack-dev=3.11.0-2 \
    bc=1.07.1-3+b1 \
    iputils-ping=3:20221126-1 \
    nano=7.2-1 \
    vim=2:9.0.1378-2 \
    git=1:2.39.2-1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Google Chrome (latest stable)
RUN wget -q -O- https://dl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-linux.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Upgrade pip and install Python requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip==23.3.1 setuptools==69.0.2 wheel==0.42.0 && \
    python -m pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data/raw_html /app/data/processed /app/models \
    /app/config /app/func /app/docker-configs /app/scripts && \
    chmod 755 /app/logs /app/data /app/models /app/config

# Copy application source code
COPY func/ /app/func/
COPY config/ /app/config/
COPY docs/ /app/docs/
COPY main.py /app/
COPY *.txt /app/
COPY *.sh /app/

# Copy Docker configuration files
COPY docker-configs/ /app/docker-configs/

# Copy supervisor configuration
COPY docker-configs/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy and install healthcheck script
COPY docker-configs/healthcheck.sh /usr/local/bin/healthcheck.sh

# Make scripts executable
RUN chmod +x /usr/local/bin/healthcheck.sh /app/docker-configs/startup.sh

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Set environment variables
ENV DISPLAY=:99 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 4000 8080

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER appuser

# Use startup script as entrypoint
ENTRYPOINT ["/app/docker-configs/startup.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
