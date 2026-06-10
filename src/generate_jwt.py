import jwt

token = jwt.encode({"sub": "user123"}, "test-secret-123", algorithm="HS256")
print(token)