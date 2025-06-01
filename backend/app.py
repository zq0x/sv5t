from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import docker
from docker.types import DeviceRequest
import time
import os
from datetime import datetime
from contextlib import asynccontextmanager
import redis.asyncio as redis

GPU_LIST = []

print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
# r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)
pool = redis.ConnectionPool(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0, decode_responses=True, max_connections=10)
r = redis.Redis(connection_pool=pool)
pipe = r.pipeline()



app = FastAPI()





                    
@app.get("/")
async def root():
    return f'Hello from server {os.getenv("BACKEND_PORT")}!'



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("BACKEND_IP")}', port=int(os.getenv("BACKEND_PORT")))