import os
import threading
import random
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, Depends
import httpx
from pymongo import MongoClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
import pathway as pw
from pipeline import Retriever
import json
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail



# Database setup
mongo = MongoClient("mongodb://mongodb:27017")
database = mongo['IITI_BOT']
users_collection = database['users']
pending_users_collection = database['pending_users']

# JWT and password hashing setup
SECRET_KEY = "Gunge Ne Gana Gaya aur Andhe Ne Movie Dekhi"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# Pathway schemas
class InputSchema(pw.Schema):
    chat_id: int
    email: str 
    query: str
    session_id: str = pw.column_definition(primary_key=True)
class HistorySchema(pw.Schema):
    userid: str = pw.column_definition(primary_key=True)
    query: str
    result: str

# Pydantic models
class UserSignup(BaseModel):
    Name : str
    email : str
    phone : str
    password : str
    member_type : str
    department : Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class OTPVerify(BaseModel):
    email: str
    otp: str

# Utility functions
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("email")
        session_id: str = payload.get("session_id")
        if email is None or session_id is None:
            raise credentials_exception
        
        return email, session_id
    except JWTError:
        raise credentials_exception

def send_otp_email(receiver_email, otp):
    message = Mail(
    from_email='khushm965@gmail.com',
    to_emails=receiver_email,
    subject='OTP FOR IITI BOT',
    html_content=f'<strong>Your OTP is {otp}</strong>')
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        # sg.set_sendgrid_data_residency("eu")
        # uncomment the above line if you are sending mail using a regional EU subuser
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)


# Pathway logic
def run_pathway():
    
    input_, output_writer = pw.io.http.rest_connector(
        webserver=pw.io.http.PathwayWebserver(host="0.0.0.0", port=8003),
        route="/ask",
        schema=InputSchema,
        delete_completed_queries=True
    )
   
    input2 = input_.select(user_id=pw.this.id, queries=pw.this.query, chat_id=pw.this.chat_id,email=pw.this.email)
    retriever = Retriever()
    output = retriever(input2)
    input_.promise_universe_is_equal_to(output)
    output = output.with_universe_of(input_)
    # input_ += output.select(result=pw.this.answer)
    
    output_writer(output.select(result=pw.this.answer))
    pw.run()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_pathway, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/signup")
async def signup(user: UserSignup):
    if users_collection.find_one({"email": user.email}) or pending_users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists or is pending verification")

    hashed_password = get_password_hash(user.password)
    otp = random.randint(100000, 999999)
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)

    user_dict = {
        "email": user.email,
        "name": user.Name,
        "phone": user.phone,
        "password": hashed_password,
        "member_type": user.member_type,
        "department": user.department,
        "otp": str(otp),
        "otp_expiry": otp_expiry
    }
    pending_users_collection.insert_one(user_dict)

    try:
        send_otp_email(user.email, otp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP: {str(e)}")

    return {"message": "OTP sent to your email. Please verify to complete registration."}

@app.post("/verify-email")
async def verify_email(data: OTPVerify):
    user = pending_users_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or already verified")

    if user["otp"] != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    if datetime.utcnow() > user["otp_expiry"]:
        raise HTTPException(status_code=400, detail="OTP expired")

    user.pop("otp")
    user.pop("otp_expiry")
    user["is_verified"] = True
    user["chats"] = []
    users_collection.insert_one(user)
    pending_users_collection.delete_one({"email": data.email})

    return {"message": "Email verified successfully!"}

@app.post("/login")
async def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not db_user.get("is_verified", False):
        raise HTTPException(status_code=403, detail="Email not verified. Please verify OTP.")

    token = create_access_token({"email": user.email,"session_id": str(random.randint(10000000000, 100000000000))})
    return {"access_token": token, "token_type": "bearer"}

@app.post('/ask')
async def ask_question(request: Request, token: str = Depends(oauth2_scheme)):
    email, session_id = get_current_user(token)
    raw_body = await request.body()

    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    body_json['email'] = email
    body_json['session_id'] = session_id
    new_body = json.dumps(body_json).encode('utf-8')

    headers = dict(request.headers)
    headers['content-length'] = str(len(new_body))
    headers['content-type'] = 'application/json'

    async with httpx.AsyncClient(timeout=300.0) as client:
        pathway_response = await client.request(
            method=request.method,
            url="http://0.0.0.0:8003/ask",
            headers={k: v for k, v in headers.items() if k.lower() != 'host'},
            content=new_body
        )
        response_headers = dict(pathway_response.headers)
        response_headers.pop("transfer-encoding", None)
        return Response(
            content=pathway_response.content,
            status_code=pathway_response.status_code,
            headers=response_headers
        )
@app.get('/history')
async def get_history(request: Request, token: str = Depends(oauth2_scheme)):
    email = get_current_user(token)
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    history = user.get('chats', {})
    return history