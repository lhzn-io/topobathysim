import argparse
import multiprocessing

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BathyServe with Parallel Workers")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9595, help="Bind port")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    args = parser.parse_args()

    # Default to CPU count if not specified, but cap at reasonable limit for laptop
    # or just use multiprocessing.cpu_count()
    if args.workers is None:
        # Default to high utilization for this task
        # User has 24 cores. Let's start with 8 to be safe or just use all?
        # Uvicorn recommends (2 x num_cores) + 1 usually, but for CPU heavy tasks, maybe simpler.
        # Let's default to a sane 8 workers if not specified, or CPU count.
        # Given user's explicit request for parallelism, let's use a good chunk.
        cpu_count = multiprocessing.cpu_count()
        # Cap default to 8 to avoid OOM if 24 processes load big datasets
        args.workers = min(cpu_count, 8)
        print(f"Auto-configured workers: {args.workers} (CPU Count: {cpu_count})")

    # If reload is True, workers must be 1 usually, or simple reload logic.
    # Uvicorn handles reload with workers=1?? No, reload excludes workers argument.
    if args.reload:
        print("Reload enabled: Forcing workers=1")
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
    else:
        print(f"Starting BathyServe on {args.host}:{args.port} with {args.workers} workers.")
        uvicorn.run("app.main:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
