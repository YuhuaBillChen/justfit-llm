"""Forward HTTP disconnects while a synchronous vision worker is preparing."""
import threading


def submit_with_disconnect(backend, job, disconnected):
    if not job.image_spans or getattr(backend, 'vision_preparer', None) is None:
        return backend.submit(job)
    stopped = threading.Event()

    def watch():
        while not stopped.wait(.05):
            try:
                gone = disconnected()
            except OSError:
                gone = True
            if gone:
                backend.cancel(job)
                return

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    try:
        return backend.submit(job)
    finally:
        stopped.set()
        monitor.join()
