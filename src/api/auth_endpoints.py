"""Authentication endpoints for the Flight Scheduler API."""

from fastapi import APIRouter, HTTPException, status, Depends, Request, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

from src.services.auth import auth_service, User, UserRole, Token
from src.services.security import (
    get_current_user, require_permission, require_role, audit_logger,
    AuditAction, get_client_ip, get_user_agent, InputValidator,
    rate_limit_by_ip, limiter
)
from src.services.auth import Permission

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class UserResponse(BaseModel):
    """User response model (without sensitive data)."""
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    permissions: List[str]


class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str
    email: EmailStr
    password: str
    role: UserRole


class UpdateUserRoleRequest(BaseModel):
    """Update user role request model."""
    username: str
    new_role: UserRole


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    timestamp: str
    action: str
    user_id: Optional[str]
    username: Optional[str]
    resource: Optional[str]
    details: dict
    ip_address: Optional[str]
    success: bool


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")  # Rate limit login attempts
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Login endpoint to authenticate users and return JWT tokens.
    
    Rate limited to 5 attempts per minute per IP address.
    """
    try:
        # Attempt login
        token = auth_service.login(form_data.username, form_data.password)
        
        if not token:
            # Log failed login attempt
            audit_logger.log_event(
                action=AuditAction.LOGIN,
                username=form_data.username,
                details={"reason": "invalid_credentials"},
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request),
                success=False
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Log successful login
        user = auth_service.get_user(form_data.username)
        audit_logger.log_event(
            action=AuditAction.LOGIN,
            user_id=user.user_id if user else None,
            username=form_data.username,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=True
        )
        
        return token
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            details={"error": str(e), "context": "login"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    refresh_token: str = Form(...)
):
    """
    Refresh access token using refresh token.
    """
    try:
        token = auth_service.refresh_access_token(refresh_token)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            details={"error": str(e), "context": "token_refresh"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Logout endpoint (client should discard tokens).
    """
    audit_logger.log_event(
        action=AuditAction.LOGOUT,
        user_id=current_user.user_id,
        username=current_user.username,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        success=True
    )
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information.
    """
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role.value,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        permissions=[p.value for p in current_user.permissions]
    )


@router.post("/users", response_model=UserResponse)
async def create_user(
    request: Request,
    user_request: CreateUserRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """
    Create a new user (admin only).
    """
    try:
        # Validate input
        username = InputValidator.sanitize_string_input(user_request.username, 50)
        
        # Create user
        new_user = auth_service.create_user(
            username=username,
            email=user_request.email,
            password=user_request.password,
            role=user_request.role
        )
        
        # Log user creation
        audit_logger.log_event(
            action=AuditAction.DATA_MODIFY,
            user_id=current_user.user_id,
            username=current_user.username,
            resource="user_creation",
            details={"created_user": new_user.username, "role": new_user.role.value},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=True
        )
        
        return UserResponse(
            user_id=new_user.user_id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role.value,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            last_login=new_user.last_login,
            permissions=[p.value for p in new_user.permissions]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            user_id=current_user.user_id,
            username=current_user.username,
            details={"error": str(e), "context": "user_creation"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )


@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    current_user: User = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """
    Get all users (admin only).
    """
    users = auth_service.get_all_users()
    return [
        UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            permissions=[p.value for p in user.permissions]
        )
        for user in users
    ]


@router.put("/users/role", response_model=UserResponse)
async def update_user_role(
    request: Request,
    role_request: UpdateUserRoleRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """
    Update user role (admin only).
    """
    try:
        updated_user = auth_service.update_user_role(
            role_request.username,
            role_request.new_role
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Log role update
        audit_logger.log_event(
            action=AuditAction.DATA_MODIFY,
            user_id=current_user.user_id,
            username=current_user.username,
            resource="user_role_update",
            details={
                "target_user": role_request.username,
                "new_role": role_request.new_role.value
            },
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=True
        )
        
        return UserResponse(
            user_id=updated_user.user_id,
            username=updated_user.username,
            email=updated_user.email,
            role=updated_user.role.value,
            is_active=updated_user.is_active,
            created_at=updated_user.created_at,
            last_login=updated_user.last_login,
            permissions=[p.value for p in updated_user.permissions]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            user_id=current_user.user_id,
            username=current_user.username,
            details={"error": str(e), "context": "role_update"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role update failed"
        )


@router.delete("/users/{username}")
async def deactivate_user(
    request: Request,
    username: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_SYSTEM))
):
    """
    Deactivate user account (admin only).
    """
    try:
        # Prevent self-deactivation
        if username == current_user.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        success = auth_service.deactivate_user(username)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Log user deactivation
        audit_logger.log_event(
            action=AuditAction.DATA_MODIFY,
            user_id=current_user.user_id,
            username=current_user.username,
            resource="user_deactivation",
            details={"deactivated_user": username},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=True
        )
        
        return {"message": f"User {username} deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            action=AuditAction.SYSTEM_ERROR,
            user_id=current_user.user_id,
            username=current_user.username,
            details={"error": str(e), "context": "user_deactivation"},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deactivation failed"
        )


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(require_permission(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get audit logs (admin only).
    """
    try:
        # Parse action filter
        action_filter = None
        if action:
            try:
                action_filter = AuditAction(action)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid action: {action}"
                )
        
        # Get logs
        logs = audit_logger.get_logs(
            user_id=user_id,
            action=action_filter,
            limit=min(limit, 1000)  # Cap at 1000 logs
        )
        
        return [
            AuditLogResponse(
                timestamp=log["timestamp"],
                action=log["action"],
                user_id=log.get("user_id"),
                username=log.get("username"),
                resource=log.get("resource"),
                details=log.get("details", {}),
                ip_address=log.get("ip_address"),
                success=log.get("success", True)
            )
            for log in logs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )