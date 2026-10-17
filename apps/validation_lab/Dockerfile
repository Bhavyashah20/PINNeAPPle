# Physics Validation Lab -- build from the repository root:
#   docker build -f apps/validation_lab/Dockerfile -t validation-lab .
#   docker run -p 8107:8107 validation-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/validation_lab/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/validation_lab
WORKDIR /srv/apps/validation_lab
EXPOSE 8107
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8107/health')"
CMD ["uvicorn", "validation_lab.api:app", "--host", "0.0.0.0", "--port", "8107", "--workers", "2"]
