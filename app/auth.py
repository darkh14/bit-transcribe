import aiohttp
from fastapi import HTTPException, Header

AUTH_SERVICE_URL = "https://devtools.1bitai.ru/9010"


async def check_token(token: str = Header()) -> bool:

    return True
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{AUTH_SERVICE_URL}/check-token", json={"token": token}, timeout=10) as response:
                if response.status == 200:
                    return True
                elif response.status == 404:
                    raise HTTPException(status_code=401, detail="Invalid token")
                elif response.status == 403:
                    raise HTTPException(status_code=401, detail="Token expired")
                else:
                    raise HTTPException(status_code=500, detail="Authentication service error")
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=500, detail=f"Error connecting to authentication service: {str(e)}")
