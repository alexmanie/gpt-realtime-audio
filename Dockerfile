FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=80

WORKDIR /app

COPY requirements.txt .
# fastrtc declares gradio<6.0 in its metadata but works fine with gradio 6.x.
# Install all other packages first, then fastrtc with --no-deps to bypass the
# stale constraint, and finally fastrtc's non-gradio dependencies explicitly.
RUN grep -v '^fastrtc' requirements.txt > /tmp/req_base.txt && \
    pip install --no-cache-dir -r /tmp/req_base.txt && \
    pip install --no-cache-dir --no-deps "fastrtc>=0.0.20" && \
    pip install --no-cache-dir aioice>=0.10.1 aiortc librosa "numba>=0.60.0" audioop-lts standard-aifc standard-sunau

COPY app.py .

EXPOSE 80

CMD ["python", "app.py"]
