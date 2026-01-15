import threading
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from concurrent.futures import ThreadPoolExecutor

# import json
import time

class NuclearExecutor(ThreadPoolExecutor):
    def shutdown(self, wait=True, **kwargs):
        print("☢️ NuclearExecutor: Abandoning stuck threads immediately.")
        super().shutdown(wait=wait, cancel_futures=True)


class ProxyServer(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8001):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._app = FastAPI()
        self._on_receive = None
        self._streaming = None
        self.server = None
        self._log_path = './trace/proxy/traffic.log'

        self.executor = NuclearExecutor(max_workers=50)

        @self._app.on_event("startup")
        async def startup_event():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(self.executor)

        @self._app.post("/")
        async def endpoint(request: Request):
            if not self._on_receive:
                return {"status": "error", "message": "Handler not set"}
            
            data = await request.json()  # dict
            # # Log data
            # with open(self._log_path, 'a') as f:
            #     f.write(f'[Client] {json.dumps(data)}\n')

            response = await self._on_receive(data)  # List[dict] if streaming, else dict
            
            # # Log response
            # with open(self._log_path, 'a') as f:
            #     if self._streaming:
            #         for line in response:
            #             f.write(f'[Server] {json.dumps(line)}\n')
            #     else:
            #         f.write(f'[Server] {json.dumps(response)}\n')

            return {"streaming_response": response} if self._streaming else response

    def set_handler(self, handler):
        """
        Set method to handle received data
        """
        self._on_receive = handler

    def set_streaming(self, streaming):
        """
        Set streaming mode
        """
        self._streaming = streaming

    def get_url(self):
        return f'http://{self._host}:{self._port}'
    
    def run(self):
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=1,
            timeout_keep_alive=1,
            loop="asyncio"
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True

        self.join(timeout=5)

        if self.is_alive():
            print("Warning: Proxy thread did not exit cleanly.")
            print("Forcing shutdown...")
            self.executor.shutdown(wait=False)


if __name__ == '__main__':
    proxy_server = ProxyServer()
    proxy_server.start()

    time.sleep(5)

    print("Check if thread is blocking.")
