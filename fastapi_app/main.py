import logging
import os
import random
import time
import asyncio
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Response, Request, BackgroundTasks
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGRPC,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterHTTP,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import inject
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

EXPOSE_PORT = os.environ.get("EXPOSE_PORT", 8000)

# otlp-grpc, otlp-http
MODE = os.environ.get("MODE", "otlp-grpc")

OTLP_GRPC_ENDPOINT = os.environ.get("OTLP_GRPC_ENDPOINT", "jaeger-collector:4317")
OTLP_HTTP_ENDPOINT = os.environ.get(
    "OTLP_HTTP_ENDPOINT", "http://jaeger-collector:4318/v1/traces"
)

TARGET_ONE_HOST = os.environ.get("TARGET_ONE_HOST", "app-b")
TARGET_TWO_HOST = os.environ.get("TARGET_TWO_HOST", "app-c")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "unknown")

app = FastAPI()


def inject_fault(service_name: str):
    # Differentiate error rates by service to create varied data for RCA
    rate = 0.3
    if service_name == "app-a":
        rate = 0.3
    elif service_name == "app-b":
        rate = 0.3
    elif service_name == "app-c":
        rate = 0.3

    if random.random() < rate:
        fault_type = random.choice(["latency", "exception", "resource"])
        if fault_type == "latency":
            delay = random.uniform(2, 7)
            logging.warning(f"[{service_name}] Fault Injection: Delaying {delay:.2f}s")
            time.sleep(delay)
        elif fault_type == "exception":
            logging.error(f"[{service_name}] Fault Injection: Raising random RuntimeError")
            raise RuntimeError(f"Simulated internal error in {service_name}")
        elif fault_type == "resource":
            logging.critical(f"[{service_name}] Fault Injection: Simulated high CPU/Memory usage")
            # Simulate high iterations
            _ = [i * i for i in range(1_000_000)]


@app.middleware("http")
async def fault_injection_middleware(request: Request, call_next):
    # Skip fault injection for metadata/root paths to keep the app "alive" for health checks
    if request.url.path not in ["/", "/docs", "/openapi.json"]:
        inject_fault(SERVICE_NAME)
    response = await call_next(request)
    return response


async def log_background_anomaly(service_name: str):
    """Simulate a background process reporting an issue."""
    await asyncio.sleep(random.uniform(1, 4))
    levels = [logging.ERROR, logging.CRITICAL, logging.WARNING]
    level = random.choice(levels)
    msg = f"[{service_name}] Background check failed: {random.choice(['Disk I/O jitter', 'Network latency spikes', 'DB connection pool near limit'])}"
    logging.log(level, msg)


def setting_jaeger(app: ASGIApp, log_correlation: bool = True) -> None:
    # set the tracer provider
    tracer = TracerProvider()
    trace.set_tracer_provider(tracer)

    if MODE == "otlp-grpc":
        tracer.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
            )
        )
    elif MODE == "otlp-http":
        tracer.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporterHTTP(endpoint=OTLP_HTTP_ENDPOINT))
        )
    else:
        # default otlp-grpc
        tracer.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporterGRPC(endpoint=OTLP_GRPC_ENDPOINT, insecure=True)
            )
        )

    # override logger format which with trace id and span id
    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)


# Setting jaeger exporter
setting_jaeger(app)


@app.get("/")
async def read_root():
    logging.error("Hello World")
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    logging.error("items")
    return {"item_id": item_id, "q": q}

'''
@app.get("/io_task")
async def io_task():
    time.sleep(1)
    logging.error("io task")
    return "IO bound task finish!"


@app.get("/cpu_task")
async def cpu_task():
    for i in range(1000):
        _ = i * i * i
    logging.error("cpu task")
    return "CPU bound task finish!"
'''

@app.get("/io_task")
async def io_task(background_tasks: BackgroundTasks):
    # Thêm: random latency spike thỉnh thoảng
    delay = random.choices([1, 3, 8, 15], weights=[30, 20, 30, 20])[0]
    if delay > 5:
        logging.critical(f"Extremely high delay detected: {delay}s")
    background_tasks.add_task(log_background_anomaly, SERVICE_NAME)
    time.sleep(delay)
    logging.error(f"io task - delay={delay}s")
    return "IO bound task finish!"


@app.get("/cpu_task")
async def cpu_task(background_tasks: BackgroundTasks):
    # Thêm: tăng workload thỉnh thoảng
    n = random.choices([1_000, 100_000, 1_000_000, 5_000_000], weights=[50, 20, 20, 10])[0]
    if n > 1_000_000:
        logging.critical(f"Heavy CPU workload: {n} iterations")
    background_tasks.add_task(log_background_anomaly, SERVICE_NAME)
    for i in range(n):
        _ = i * i * i
    logging.error(f"cpu task - iterations={n}")
    return "CPU bound task finish!"

@app.get("/random_status")
async def random_status(response: Response):
    response.status_code = random.choice([200, 200, 300, 400, 500])
    logging.error("random status")
    return {"path": "/random_status"}


@app.get("/random_sleep")
async def random_sleep(response: Response):
    time.sleep(random.randint(0, 5))
    logging.error("random sleep")
    return {"path": "/random_sleep"}


@app.get("/error_test")
async def error_test(response: Response):
    error_type = random.choice(["Value", "Key", "Attribute", "ZeroDivision"])
    logging.error(f"Triggering {error_type} error test")
    if error_type == "Value":
        raise ValueError("Simulated Value Error")
    elif error_type == "Key":
        _ = {}["missing_key"]
    elif error_type == "Attribute":
        _ = None.attribute
    else:
        _ = 1 / 0


@app.get("/chain")
async def chain(response: Response, background_tasks: BackgroundTasks):
    headers = {}
    inject(headers)  # inject trace info to header
    logging.critical(f"Chain started on {SERVICE_NAME} with headers: {headers}")
    background_tasks.add_task(log_background_anomaly, SERVICE_NAME)

    async with httpx.AsyncClient() as client:
        await client.get(
            "http://localhost:8000/",
            headers=headers,
        )
    async with httpx.AsyncClient() as client:
        await client.get(
            f"http://{TARGET_ONE_HOST}:8000/io_task",
            headers=headers,
        )
    async with httpx.AsyncClient() as client:
        await client.get(
            f"http://{TARGET_TWO_HOST}:8000/cpu_task",
            headers=headers,
        )
    logging.info("Chain Finished")
    return {"path": "/chain"}


if __name__ == "__main__":
    # update uvicorn access logger format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=EXPOSE_PORT, log_config=log_config)
