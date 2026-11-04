FROM python:3.12-slim

# xgboost needs the OpenMP runtime on Linux
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Render sets $PORT (10000 by default)
EXPOSE 10000
CMD ["sh", "-c", "streamlit run DiaCDSS_Showcase.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none"]
