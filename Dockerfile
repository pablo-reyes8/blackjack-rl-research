FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

CMD ["blackjack-train", "--experiment-config", "configs/experiments/smoke-test.yaml"]
