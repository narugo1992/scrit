# pyskeb

A Python client library for interacting with Skeb.jp, the commission platform for artists and creators.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pyskeb import SkebClient

# Initialize the client
client = SkebClient()

# Fetch artwork posts
for post in client.iter_art_pages():
    print(f"Post ID: {post['id']}")
    print(f"Creator: {post['creator']['name']}")
```

## Features

- Browse artwork posts from Skeb.jp
- Retrieve user profiles and portfolios
- Access work details and metadata
- Built-in rate limiting and retry logic
- Automatic session management

## API Reference

### SkebClient

The main client class for interacting with Skeb.jp.

#### Methods

- `iter_art_pages()` - Iterate through artwork posts
- `get_post(post_id)` - Retrieve a specific post by ID
- `iter_user_pages(user_id)` - Iterate through a user's posts
- `iter_work_pages()` - Iterate through work listings

## Configuration

The client handles authentication and rate limiting automatically. For advanced usage, you can configure custom session parameters:

```python
from pyskeb.utils import get_requests_session

session = get_requests_session()
client = SkebClient(session=session)
```

## Requirements

- Python 3.8+
- requests
- hbutils

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is an unofficial client library. Please respect Skeb.jp's terms of service and rate limits when using this library.
