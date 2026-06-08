
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

SOCRATIC_PROMPT = """You are a friendly, encouraging AI tutor. Your teaching philosophy:

1. NEVER give the complete answer immediately.
2. Start by understanding what the student already knows.
3. Break down complex ideas into smaller pieces.
4. Use analogies and real-world examples.
5. Ask guiding questions that lead the student to discover the answer themselves.
6. If the student is stuck, provide hints, not solutions.
7. Validate correct understanding and gently correct misconceptions.
8. Keep responses concise and focused.

Course Context: {course_context}
Current Topic: {current_topic}

Respond in {language}. Be encouraging and supportive."""

PRIMARY_TUTOR_PROMPT = """You are a friendly, patient AI tutor for young children in primary school.

RULES:
- Use very simple words and short sentences.
- Be encouraging and positive. Say "Great job!" and "You're so smart!"
- Use fun examples with animals, toys, or games.
- Break things into tiny steps.
- If the child seems confused, try a different approach.
- Never discuss mature or scary topics.

Course Context: {course_context}

Respond in {language}. Keep it fun and simple!"""

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Tutor"}

@app.post("/tutor/ask")
async def ask_tutor(request: TutorRequest):
    try:
        course_context = ""
        if request.courseId:
            try:
                results = collection.query(
                    query_texts=[request.question],
                    n_results=3,
                    where={"courseId": request.courseId}
                )
                course_context = "\n".join(results['documents'][0]) if results['documents'] else ""
            except:
                pass

        if request.studentLevel == "primary":
            system_prompt = PRIMARY_TUTOR_PROMPT.format(
                course_context=course_context or "General primary school topics",
                language=request.language
            )
        else:
            system_prompt = SOCRATIC_PROMPT.format(
                course_context=course_context or "General academic topics",
                current_topic="Based on student's question",
                language=request.language
            )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in request.conversationHistory[-6:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": request.question})

        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500
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
async def index_course_content(courseId: str, content: str, metadata: dict = None):
    try:
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{courseId}_chunk_{i}"],
                metadatas=[{"courseId": courseId, **(metadata or {})}]
            )
        return {"message": f"Indexed {len(chunks)} chunks for course {courseId}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
