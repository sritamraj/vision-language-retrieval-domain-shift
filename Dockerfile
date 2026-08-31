# Hugging Face Docker Spaces build for the research demo (app/demo.py).
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker
#
# Spaces routes public traffic to port 7860 by default (overridable via
# `app_port` in the Space README frontmatter). Spaces also runs containers
# as an arbitrary non-root UID, so we create a `user` and chown everything
# up front rather than relying on root-owned files.

FROM python:3.10-slim

# Minimal system deps for Pillow/torch image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

USER user

EXPOSE 7860

CMD ["python", "app/demo.py"]
