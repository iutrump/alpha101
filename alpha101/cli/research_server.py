from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("alpha101.research.server:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    main()
