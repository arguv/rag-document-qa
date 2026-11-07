"""ASGI boundary: bounded bodies, API-key access, request IDs and process metrics."""
import hmac
import json
import logging
import os
import time
from collections import Counter, deque
from uuid import uuid4

logger = logging.getLogger('rag.requests')


class BoundaryMiddleware:
    def __init__(self, app):
        self.app = app
        self.windows = {}  # per client, bounded by MAX_RATE_CLIENTS
        self.counts = Counter()
        self.duration = 0.0
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        headers = dict(scope['headers'])
        path = scope['path']
        started = time.monotonic()
        status = 500
        response_started = False

        async def respond(code, detail):
            nonlocal status
            status = code
            await send({'type': 'http.response.start', 'status': code, 'headers': [(b'content-type', b'application/json'), (b'x-request-id', request_id.encode())]})
            await send({'type': 'http.response.body', 'body': json.dumps({'detail': detail}).encode()})

        async def tracked(message):
            nonlocal status, response_started
            if message['type'] == 'http.response.start':
                response_started = True
                status = message['status']
                message['headers'] = list(message.get('headers', [])) + [(b'x-request-id', request_id.encode()), (b'x-content-type-options', b'nosniff')]
            await send(message)

        try:
            key = os.getenv('API_KEY', '')
            if os.getenv('APP_ENV', 'development') == 'production' and not key:
                return await respond(503, 'API_KEY is required in production')
            if path not in ('/health', '/swagger', '/json.json') and key and not hmac.compare_digest(headers.get(b'x-api-key', b''), key.encode()):
                return await respond(401, 'Invalid API key')
            if path != '/health':
                client = (scope.get('client') or ('unknown',))[0]
                if client not in self.windows:
                    if len(self.windows) >= 10000:
                        expired = [c for c, w in self.windows.items() if not w or w[-1] <= started - 60]
                        for c in expired:
                            del self.windows[c]
                        if len(self.windows) >= 10000:
                            return await respond(429, 'Rate limiter capacity reached')
                    self.windows[client] = deque()
                window = self.windows[client]
                while window and window[0] <= started - 60:
                    window.popleft()
                if len(window) >= int(os.getenv('RATE_LIMIT_PER_MINUTE', '60')):
                    return await respond(429, 'Rate limit exceeded')
                window.append(started)
            if self.active >= int(os.getenv('MAX_CONCURRENT_REQUESTS', '8')):
                return await respond(503, 'Server busy')
            if path == '/metrics':
                body = '\n'.join(f'rag_http_requests_total{{status="{code}"}} {count}' for code, count in sorted(self.counts.items()))
                body += f'\nrag_http_duration_seconds_sum {self.duration}\nrag_http_duration_seconds_count {sum(self.counts.values())}\n'
                await tracked({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'text/plain; version=0.0.4')]})
                return await send({'type': 'http.response.body', 'body': body.encode()})
            self.active += 1
            try:
                # Enforce before multipart parsing, including chunked requests.
                cap = int(os.getenv('MAX_UPLOAD_BYTES', '10485760')) + 65536 if path == '/documents' else 65536
                data = bytearray()
                while True:
                    event = await receive()
                    if event['type'] == 'http.disconnect':
                        return
                    chunk = event.get('body', b'')
                    if len(data) + len(chunk) > cap:
                        return await respond(413, 'Request body too large')
                    data.extend(chunk)
                    if not event.get('more_body', False):
                        break
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {'type': 'http.request', 'body': bytes(data), 'more_body': False}
                    return await receive()

                await self.app(scope, replay, tracked)
            finally:
                self.active -= 1
        except Exception:
            # Avoid logging document contents, credentials or provider error payloads.
            logger.error('request_failed request_id=%s', request_id)
            if not response_started:
                await respond(503, 'Service unavailable')
        finally:
            duration = time.monotonic() - started
            self.counts[status] += 1
            self.duration += duration
            logger.info('request_complete request_id=%s status=%s duration=%.3f', request_id, status, duration)
