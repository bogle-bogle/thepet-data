from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def test():
    response = 'GET 메서드 실행 완료'
    return {response}


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
