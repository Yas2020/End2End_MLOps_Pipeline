# Logs will be JSON — great for scraping, parsing, monitoring!
import logging
import sys
from pythonjsonlogger import jsonlogger
from opentelemetry import trace

class OTELTraceIdFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            record.trace_id = format(span.get_span_context().trace_id, '032x')
            record.span_id = format(span.get_span_context().span_id, '016x')
        else:
            record.trace_id = None
            record.span_id = None
        return True

def get_logger(name="fraud-inference"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logHandler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s span_id=%(span_id)s %(message)s'
        )
        logHandler.setFormatter(formatter)
        logger.addHandler(logHandler)
        logger.setLevel(logging.INFO)
    
    if not any(isinstance(f, OTELTraceIdFilter) for f in logger.filters):
        logger.addFilter(OTELTraceIdFilter())

    return logger
