import api.routes as routes
import api.utils.settings as settings

import uvicorn
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware

settings = settings.Settings()
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
)

@app.get('/')
def read_root():
    return {'Hello': 'World'}

app.include_router(routes.slices.router, prefix='/v1')
app.include_router(routes.cluster.router, prefix='/cluster')

if __name__ == '__main__':
    uvicorn.run('server:app', host='localhost', port=settings.port, reload=True)
