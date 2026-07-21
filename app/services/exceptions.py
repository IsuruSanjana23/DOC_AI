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