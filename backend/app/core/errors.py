import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def not_found(message: str = "We couldn't find what you were looking for.") -> AppError:
    return AppError(status.HTTP_404_NOT_FOUND, "not_found", message)


def error_body(code: str, message: str, details: dict | None = None) -> dict:
    body: dict = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


_HTTP_MESSAGES = {
    404: ("not_found", "This endpoint doesn't exist."),
    405: ("method_not_allowed", "This action isn't supported here."),
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields: dict[str, str] = {}
        for err in exc.errors():
            location = [str(p) for p in err.get("loc", []) if p not in ("body", "query", "path")]
            key = ".".join(location) or "request"
            message = str(err.get("msg", "Invalid value.")).removeprefix("Value error, ")
            fields.setdefault(key, message[0].upper() + message[1:] if message else "Invalid value.")
        return JSONResponse(
            error_body("validation_error", "Some fields need your attention.", {"fields": fields}),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = _HTTP_MESSAGES.get(exc.status_code, ("http_error", str(exc.detail)))
        return JSONResponse(error_body(code, message), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path, "method": request.method})
        return JSONResponse(
            error_body("internal_error", "Something went wrong on our side. Try again in a moment."),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
