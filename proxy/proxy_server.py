import threading
import uvicorn
from fastapi import FastAPI, Request

import json
import time

class ProxyServer(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8001):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._app = FastAPI()
        self._on_receive = []  # Must be a list to allow updating handler without restarting proxy
        self.server = None

        @self._app.post("/")
        async def endpoint(request: Request):
            if not self._on_receive:
                return {"status": "error", "message": "handler not set"}
            
            data = await request.json()  # dict
            # Log data
            with open('./proxy/traffic.log', 'a') as f:
                f.write(f'[Client] {json.dumps(data)}\n')

            streaming = data.get("stream", False)
            response = await self._on_receive[0](data, streaming)  # List[dict] if streaming, else dict
            
            # Log response
            with open('./proxy/traffic.log', 'a') as f:
                if streaming:
                    for line in response:
                        f.write(f'[Server] {json.dumps(line)}\n')
                else:
                    f.write(f'[Server] {json.dumps(response)}\n')

            return {"full_response": response} if streaming else response

    def set_handler(self, handler):
        """
        Set method to handle received data
        """
        if not self._on_receive:
            self._on_receive.append(handler)
        else:
            self._on_receive[0] = handler

    def get_url(self):
        return f'http://{self._host}:{self._port}'
    
    def run(self):
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            access_log=False
        )
        self.server = uvicorn.Server(config)
        self.server.run()


if __name__ == '__main__':
    proxy_server = ProxyServer()
    proxy_server.start()

    time.sleep(5)

    print("Check if thread is blocking.")
