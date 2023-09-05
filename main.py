from fastapi import FastAPI

app = FastAPI()

@app.post("/")
async def abc():
    response = ''
    return {response}


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
