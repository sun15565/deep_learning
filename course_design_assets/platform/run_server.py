from app import _init_db, app


if __name__ == "__main__":
    _init_db()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
