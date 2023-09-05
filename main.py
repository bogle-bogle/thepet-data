from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def abc():
    response = ''
    print('abc')
    return {response}


# FastAPI 앱 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
