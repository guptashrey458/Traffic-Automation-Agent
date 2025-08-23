"""Authentication and authorization service."""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class UserRole(str, Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions."""
    READ_FLIGHTS = "read:flights"
    WRITE_FLIGHTS = "write:flights"
    READ_ANALYTICS = "read:analytics"
    OPTIMIZE_SCHEDULE = "optimize:schedule"
    MANAGE_ALERTS = "manage:alerts"
    ADMIN_SYSTEM = "admin:system"
    VIEW_AUDIT_LOGS = "view:audit_logs"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_FLIGHTS,
        Permission.WRITE_FLIGHTS,
        Permission.READ_ANALYTICS,
        Permission.OPTIMIZE_SCHEDULE,
        Permission.MANAGE_ALERTS,
        Permission.ADMIN_SYSTEM,
        Permission.VIEW_AUDIT_LOGS,
    ],
    UserRole.OPERATOR: [
        Permission.READ_FLIGHTS,
        Permission.WRITE_FLIGHTS,
        Permission.READ_ANALYTICS,
        Permission.OPTIMIZE_SCHEDULE,
        Permission.MANAGE_ALERTS,
    ],
    UserRole.ANALYST: [
        Permission.READ_FLIGHTS,
        Permission.READ_ANALYTICS,
        Permission.OPTIMIZE_SCHEDULE,
    ],
    UserRole.VIEWER: [
        Permission.READ_FLIGHTS,
        Permission.READ_ANALYTICS,
    ],
}


class User(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[Permission] = []

    def __init__(self, **data):
        super().__init__(**data)
        # Set permissions based on role
        self.permissions = ROLE_PERMISSIONS.get(self.role, [])


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = []


class AuthService:
    """Authentication and authorization service."""

    def __init__(self):
        self.users_db: Dict[str, UserInDB] = {}
        self._initialize_default_users()

    def _initialize_default_users(self):
        """Initialize default users for development."""
        default_users = [
            {
                "user_id": "admin_001",
                "username": "admin",
                "email": "admin@flightscheduler.com",
                "role": UserRole.ADMIN,
                "password": "admin123",  # Change in production
            },
            {
                "user_id": "operator_001",
                "username": "operator",
                "email": "operator@flightscheduler.com",
                "role": UserRole.OPERATOR,
                "password": "operator123",
            },
            {
                "user_id": "analyst_001",
                "username": "analyst",
                "email": "analyst@flightscheduler.com",
                "role": UserRole.ANALYST,
                "password": "analyst123",
            },
        ]

        for user_data in default_users:
            password = user_data.pop("password")
            user = User(**user_data, created_at=datetime.now(timezone.utc))
            self.users_db[user.username] = UserInDB(
                **user.model_dump(),
                hashed_password=self.get_password_hash(password)
            )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self.users_db.get(username)

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        return user

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")
            permissions: List[str] = payload.get("permissions", [])
            
            if username is None:
                return None
            
            token_data = TokenData(
                username=username,
                user_id=user_id,
                role=role,
                permissions=permissions
            )
            return token_data
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None

    def login(self, username: str, password: str) -> Optional[Token]:
        """Login user and return tokens."""
        user = self.authenticate_user(username, password)
        if not user:
            return None

        # Create token payload
        token_data = {
            "sub": user.username,
            "user_id": user.user_id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions]
        }

        # Create tokens
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token({"sub": user.username, "user_id": user.user_id})

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    def refresh_access_token(self, refresh_token: str) -> Optional[Token]:
        """Refresh access token using refresh token."""
        token_data = self.verify_token(refresh_token, "refresh")
        if not token_data or not token_data.username:
            return None

        user = self.get_user(token_data.username)
        if not user or not user.is_active:
            return None

        # Create new access token
        new_token_data = {
            "sub": user.username,
            "user_id": user.user_id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions]
        }

        access_token = self.create_access_token(new_token_data)

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,  # Keep the same refresh token
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions

    def require_permission(self, user: User, permission: Permission) -> None:
        """Raise exception if user doesn't have permission."""
        if not self.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )

    def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create a new user."""
        if username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        user_id = f"{role.value}_{len(self.users_db) + 1:03d}"
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=datetime.now(timezone.utc)
        )

        self.users_db[username] = UserInDB(
            **user.model_dump(),
            hashed_password=self.get_password_hash(password)
        )

        return user

    def update_user_role(self, username: str, new_role: UserRole) -> Optional[User]:
        """Update user role and permissions."""
        user = self.users_db.get(username)
        if not user:
            return None

        user.role = new_role
        user.permissions = ROLE_PERMISSIONS.get(new_role, [])
        return User(**user.model_dump())

    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account."""
        user = self.users_db.get(username)
        if not user:
            return False

        user.is_active = False
        return True

    def get_all_users(self) -> List[User]:
        """Get all users (admin only)."""
        return [User(**user.model_dump()) for user in self.users_db.values()]


# Global auth service instance
auth_service = AuthService()