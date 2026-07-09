"""Start the Flask REST API server."""

from api.app import create_app

if __name__ == "__main__":
    import os

    app = create_app()
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
