class GenericException(Exception):
    """Base class of every expected failure raised by the archiving flow."""


class ResourceGone(GenericException):
    """The resource was deleted, expired or made private. Retrying will not help."""


class NoContent(ResourceGone):
    """The link is alive but carries nothing worth archiving (text-only post, empty folder, ...)."""


class ResourceBlocked(GenericException):
    """The host refused us for now (quota, anti-bot page). Retry after a cool-down."""

    def __init__(self, message: str, cooldown: float = 25 * 60.0):
        super().__init__(message)
        self.cooldown = cooldown


class ResourceTransient(GenericException):
    """Network failure, 5xx or truncated body. Retry later."""


class UnexpectedResponse(GenericException):
    """An HTTP response that does not look like the file we asked for."""

    def __init__(self, status: int, content_type: str = '', snippet: str = '', url: str = ''):
        super().__init__(f'HTTP {status} {content_type!r} for {url!r}: {snippet[:160]!r}')
        self.status = status
        self.content_type = content_type
        self.snippet = snippet
        self.url = url
