class DuplicateEmailException(Exception):
    pass


class CredentialsException(Exception):
    pass


class NotFoundException(Exception):
    pass


class DuplicateCollectionNameException(Exception):
    pass


class InvalidFileTypeException(Exception):
    pass


class FileTooLargeException(Exception):
    pass

class ExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


class PromptBuilderError(Exception):
    """Raised when prompt construction fails.

    Covers empty queries, malformed search results, and any other
    input validation failure within the prompt builder.
    """
    pass


class LLMServiceError(Exception):
    """Raised when LLM communication fails.

    Covers API errors, network timeouts, rate limits, empty or
    malformed responses, and configuration errors.
    """
    pass