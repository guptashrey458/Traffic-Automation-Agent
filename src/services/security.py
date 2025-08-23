"""Security middleware and utilities."""

import os
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cryptography.fernet import Fernet
import logging

from .auth import auth_service, User, Permission

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Data encryption
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

# HTTP Bearer token scheme
security = HTTPBearer()


class SecurityLevel(str, Enum):
    """Security levels for different data types."""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"


class AuditAction(str, Enum):
    """Audit log action types."""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFY = "data_modify"
    OPTIMIZATION_RUN = "optimization_run"
    ALERT_GENERATED = "alert_generated"
    ALERT_RESOLVED = "alert_resolved"
    SYSTEM_ERROR = "system_error"


class AuditLog:
    """Audit logging service."""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
    
    def log_event(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ):
        """Log an audit event."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action.value,
            "user_id": user_id,
            "username": username,
            "resource": resource,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success
        }
        
        self.logs.append(log_entry)
        
        # Log to structured logger
        logger.info(
            f"Audit: {action.value}",
            extra={
                "audit": True,
                "user_id": user_id,
                "username": username,
                "resource": resource,
                "success": success,
                "ip_address": ip_address
            }
        )
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filtering."""
        filtered_logs = self.logs.copy()
        
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.get("user_id") == user_id]
        
        if action:
            filtered_logs = [log for log in filtered_logs if log.get("action") == action.value]
        
        if start_time:
            start_iso = start_time.isoformat()
            filtered_logs = [log for log in filtered_logs if log.get("timestamp", "") >= start_iso]
        
        if end_time:
            end_iso = end_time.isoformat()
            filtered_logs = [log for log in filtered_logs if log.get("timestamp", "") <= end_iso]
        
        # Sort by timestamp (newest first) and limit
        filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return filtered_logs[:limit]


# Global audit logger
audit_logger = AuditLog()


class DataEncryption:
    """Data encryption utilities."""
    
    @staticmethod
    def encrypt_sensitive_data(data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted_data = cipher_suite.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Data encryption failed"
            )
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Data decryption failed"
            )
    
    @staticmethod
    def hash_sensitive_field(data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive field for storage."""
        if salt is None:
            salt = os.getenv("HASH_SALT", "default_salt")
        
        combined = f"{data}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers (behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """Get user agent from request."""
    return request.headers.get("User-Agent", "unknown")


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        token_data = auth_service.verify_token(credentials.credentials)
        if token_data is None or token_data.username is None:
            audit_logger.log_event(
                action=AuditAction.ACCESS_DENIED,
                details={"reason": "invalid_token"},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            raise credentials_exception
        
        # Get user
        user_in_db = auth_service.get_user(token_data.username)
        if user_in_db is None:
            audit_logger.log_event(
                action=AuditAction.ACCESS_DENIED,
                username=token_data.username,
                details={"reason": "user_not_found"},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            raise credentials_exception
        
        if not user_in_db.is_active:
            audit_logger.log_event(
                action=AuditAction.ACCESS_DENIED,
                user_id=user_in_db.user_id,
                username=user_in_db.username,
                details={"reason": "user_inactive"},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        # Convert to User model
        user = User(**user_in_db.model_dump())
        
        # Log successful access
        audit_logger.log_event(
            action=AuditAction.DATA_ACCESS,
            user_id=user.user_id,
            username=user.username,
            resource="api_access",
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=True
        )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            details={"error": str(e), "context": "authentication"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise credentials_exception


def require_permission(permission: Permission):
    """Dependency to require specific permission."""
    def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not auth_service.has_permission(current_user, permission):
            audit_logger.log_event(
                action=AuditAction.ACCESS_DENIED,
                user_id=current_user.user_id,
                username=current_user.username,
                details={"required_permission": permission.value},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    
    return permission_checker


def require_role(allowed_roles: List[str]):
    """Dependency to require specific roles."""
    def role_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role.value not in allowed_roles:
            audit_logger.log_event(
                action=AuditAction.ACCESS_DENIED,
                user_id=current_user.user_id,
                username=current_user.username,
                details={"required_roles": allowed_roles, "user_role": current_user.role.value},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: one of {allowed_roles}"
            )
        return current_user
    
    return role_checker


class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_airport_code(airport: str) -> str:
        """Validate airport code format."""
        if not airport or len(airport) != 3 or not airport.isalpha():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid airport code format (must be 3 letters)"
            )
        return airport.upper()
    
    @staticmethod
    def validate_flight_id(flight_id: str) -> str:
        """Validate flight ID format."""
        if not flight_id or len(flight_id) < 3 or len(flight_id) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid flight ID format"
            )
        return flight_id.upper()
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> None:
        """Validate date range."""
        if start_date >= end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Limit date range to prevent excessive queries
        max_days = 90
        if (end_date - start_date).days > max_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Date range cannot exceed {max_days} days"
            )
    
    @staticmethod
    def sanitize_string_input(input_str: str, max_length: int = 255) -> str:
        """Sanitize string input."""
        if not input_str:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = input_str.strip()
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature for secure integrations."""
    if not signature or not secret:
        return False
    
    try:
        # Remove 'sha256=' prefix if present
        if signature.startswith('sha256='):
            signature = signature[7:]
        
        # Calculate expected signature
        expected_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Secure comparison
        return hmac.compare_digest(signature, expected_signature)
    
    except Exception as e:
        logger.error(f"Webhook signature verification failed: {e}")
        return False


# Security middleware functions
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    
    return response


# Rate limiting decorators
def rate_limit_by_user(requests_per_minute: int = 60):
    """Rate limit by authenticated user."""
    def decorator(func):
        return limiter.limit(f"{requests_per_minute}/minute")(func)
    return decorator


def rate_limit_by_ip(requests_per_minute: int = 30):
    """Rate limit by IP address."""
    def decorator(func):
        return limiter.limit(f"{requests_per_minute}/minute")(func)
    return decorator