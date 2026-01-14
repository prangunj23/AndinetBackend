import json
import os

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from cerebras.cloud.sdk import Cerebras

# ======================================================
# ENV
# ======================================================
load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY not found in .env")

cerebras = Cerebras(api_key=CEREBRAS_API_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# ENDPOINT
# ======================================================
@app.get("/process")
def process(text: str = Query(..., description="Prompt to run on")):

    def event_generator():
        yield f"data: {json.dumps({'status': 'Creating outline'})}\n\n"

        PROMPT = text

        AGENTS = {
            "LinkedIn": "Search and recruit candidates using LinkedIn Recruiter-style natural language queries.",
            "Gemini": "Generate supporting content, summaries, and outreach messaging.",
            "SimpleAI(S24)": "Handle automated outreach or follow-up communication."
        }

        FORMAT = """
            ```json
            [
            {
                "agent": "string",
                "label": "string",
                "prompt": "string",
                "url": "string"
            }
            ]
        """
        LLM_PROMPT = (
            f"Break {PROMPT} into smaller tasks to be handled by {AGENTS}. "
            f"Be as specific as possible, and mention exactly what each agent should do in detailed steps. "
            f"An example is step 1: navigate to (sample.url), step 2: search for X, etc. "
            f"For each task, return an object with:\n"
            f"- agent\n"
            f"- prompt\n"
            f"- label\n"
            f"- url (the primary website the agent should operate on)\n\n"
            f"Rules for url:\n"
            f"- LinkedIn → https://www.linkedin.com/\n"
            f"- Gemini / Google tools → https://workspace.google.com/\n"
            f"- SimpleAI(S24) → https://www.simple.ai/ (or company homepage)\n\n"
            f"The information produced by one agent can go into another.\n"
            f"Format the output exactly like this: {FORMAT}.\n"
            f"Do not include anything else."
        )

        response = cerebras.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": LLM_PROMPT}],
            max_tokens=2000,
            temperature=0.7,
        )

        output = response.choices[0].message.content

        try:
            # --------------------------------------------------
            # Parse JSON safely
            # --------------------------------------------------
            if "```" in output:
                output = output.split("```")[1]

            output = output.replace("json", "").strip()
            data = json.loads(output)

            n = len(data)

            yield f"data: {json.dumps({'status': f'{n} tasks generated'})}\n\n"

            # --------------------------------------------------
            # Run each agent (simulation only)
            # --------------------------------------------------
            for i, task in enumerate(data):
                agent_name = task["agent"]
                agent_prompt = task["prompt"]
                agent_url = task["url"]
                agent_desc = AGENTS[agent_name]

                yield f"data: {json.dumps({'progress': f'{i+1}/{n}', 'agent': agent_name})}\n\n"

                agent_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"You are a {agent_name} agent.\n\n"
                            f"Capabilities:\n{agent_desc}\n\n"
                            f"Website:\n{agent_url}\n\n"
                            f"Task:\n{agent_prompt}\n\n"
                            "Produce a realistic sample output."
                        )
                    }
                ]

                agent_response = cerebras.chat.completions.create(
                    model="gpt-oss-120b",
                    messages=agent_messages,
                    max_tokens=2000,
                    temperature=0.7,
                )

                task["output"] = agent_response.choices[0].message.content

            # --------------------------------------------------
            # Done
            # --------------------------------------------------
            yield f"data: {json.dumps({'progress': 'done', 'result': data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'raw_output': output})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class ChatRequest(BaseModel):
    message: str
    context: str | None = None

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
                    You are a helpful pipeline assistant.

                    Context:
                    {req.context or "None"}

                    User:
                    {req.message}
                    """,
    )
    return {"message": response.text}