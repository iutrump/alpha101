from alpha101.research.server import app, get_context, run_backtest


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)
