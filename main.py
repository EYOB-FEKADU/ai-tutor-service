from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Tutor Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="course_content")

class TutorRequest(BaseModel):
    question: str
    courseId: str = None
    lessonId: str = None
    studentLevel: str = "highschool"
    language: str = "en"
    conversationHistory: list = []

class IndexRequest(BaseModel):
    courseId: str
    content: str
    metadata: dict = None

SOCRATIC_PROMPT = """You are a helpful, knowledgeable AI tutor integrated into a learning platform.

Your teaching approach:
1. Give a clear, detailed explanation first — don't just ask questions.
2. Break down complex topics into simple, digestible parts.
3. Use examples, analogies, and real-world applications.
4. After explaining, ask ONE follow-up question to check understanding.
5. If the student asks for more detail, go deeper.
6. If the student seems confused, simplify and try a different angle.
7. Be encouraging and supportive.
8. Keep responses focused and relevant to the course material.

Course Context: {course_context}

Respond in {language}. Be thorough but not overwhelming."""

PRIMARY_TUTOR_PROMPT = """You are a friendly, patient AI tutor for young children.

RULES:
- Use very simple words and short sentences.
- Give clear, simple explanations with fun examples.
- Use emojis and encouraging words like "Great job!" and "You're so smart!"
- After explaining, ask one simple question.
- If the child seems confused, use a different example.
- Never discuss mature or scary topics.

Course Context: {course_context}

Respond in {language}. Keep it fun and simple!"""

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Tutor"}

@app.post("/tutor/ask")
async def ask_tutor(request: TutorRequest):
    try:
        course_context = "General academic topics"
        if request.courseId:
            try:
                results = collection.query(
                    query_texts=[request.question],
                    n_results=3,
                    where={"courseId": request.courseId}
                )
                if results['documents'] and results['documents'][0]:
                    course_context = "\n".join(results['documents'][0])
            except:
                pass

        if request.studentLevel == "primary":
            system_prompt = PRIMARY_TUTOR_PROMPT.format(
                course_context=course_context,
                language=request.language
            )
        else:
            system_prompt = SOCRATIC_PROMPT.format(
                course_context=course_context,
                language=request.language
            )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in request.conversationHistory[-10:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": request.question})

        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )

        reply = response.choices[0].message.content

        return {
            "response": reply,
            "model": "llama-3.1-8b-instant",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@app.post("/tutor/index-course")
async def index_course_content(request: IndexRequest):
    try:
        chunks = [request.content[i:i+500] for i in range(0, len(request.content), 500)]
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{request.courseId}_chunk_{i}"],
                metadatas=[{"courseId": request.courseId, **(request.metadata or {})}]
            )
        return {"message": f"Indexed {len(chunks)} chunks for course {request.courseId}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
