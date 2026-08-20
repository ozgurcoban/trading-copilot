"""Domain-specific errors for snapshot construction."""


class TradingCopilotError(Exception):
    """Base error for the package."""


class InvalidTickerError(TradingCopilotError):
    """The requested ticker has an unsupported shape."""


class InvalidAsOfError(TradingCopilotError):
    """The requested as-of date is invalid."""


class MarketDataFetchError(TradingCopilotError):
    """Market data could not be fetched."""


class MarketDataValidationError(TradingCopilotError):
    """Market data failed deterministic validation."""


class InsufficientHistoryError(MarketDataValidationError):
    """There are too few valid sessions to build the snapshot."""
