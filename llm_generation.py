import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from huggingface_hub import login, InferenceClient


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],         # or ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

login(os.getenv("HUGGINGFACE_KEY"))

class Input(BaseModel):
    text: str

@app.get("/process")
def process(text: str = Query(..., description="Prompt to run on")):
    def event_generator():
        print("running")
        yield f"data: {json.dumps({"status": "Creating Outline"})}\n\n"
        PROMPT = text
        AGENTS = '''{"LinkedIn": "The AI-Assisted Search feature in LinkedIn Recruiter leverages Generative AI (GAI) to streamline and enhance the talent discovery process. Instead of relying solely on traditional keyword filters and Boolean strings, recruiters can now describe in natural language what kind of candidate they’re looking for—such as specific skills, experience levels, or job histories—and the AI automatically interprets and generates a targeted search query. This allows LinkedIn Recruiter to deliver a refined list of candidates who closely match the desired profile, even if their résumés use different terminology. The system learns from recruiter interactions and feedback to continuously improve search precision over time. In addition, it provides context-aware recommendations, highlights transferable skills, and surfaces diverse talent pools that might have been missed through conventional searches. Overall, AI-Assisted Search helps recruiters save time, reduce manual search complexity, and discover high-quality candidates more efficiently while maintaining a human-centered approach to final selection and outreach.",
            "Gemini": "Content creation tools enable users to generate and refine various types of materials efficiently. For writing, AI can produce first drafts for emails and documents, help brainstorm ideas, and refine content for clarity and tone. Presentations can be transformed from simple ideas into polished slides with AI-generated visuals. In spreadsheets, AI assists with organizing and processing data, creating templates, and automatically filling in missing values. It can even generate video content from text prompts or images, creating cinematic scenes or short clips. In information and data analysis, AI can automatically summarize long documents, meetings, and message threads, analyze raw data to provide insights and formatting suggestions, and use research tools like NotebookLM to extract instant insights and summaries from uploaded sources. It can also perform intelligent searches across platforms like Google Drive by understanding user intent and context. For collaboration and communication, AI can generate meeting agendas, transcribe conversations in real time, and summarize discussions with clear action items. It provides real-time translation to support international teams and automates workflows across Google products—such as transcribing meeting notes and turning them into checklists. Additional capabilities include enhanced communication features that suggest better phrasing, flag misunderstandings, and prioritize important messages, as well as customer service support that helps build customer profiles and draft responses efficiently.",
            "SimpleAI(S24)": "Simple AI is an advanced conversational platform designed to automate and enhance voice-based customer interactions using artificial intelligence. It functions as a virtual phone agent that can make and receive calls, understand natural language, and respond intelligently based on real-time context and company data. When a call is triggered—either through a CRM system, an API, or automated scheduling—Simple AI pulls relevant customer information, such as name, account details, and previous interactions, to personalize the conversation. The AI then engages the caller naturally, handling inquiries, objections, or routine transactions with human-like dialogue. It can access your company’s knowledge base in real time to provide accurate information, support multiple languages, and even navigate complex IVR menus. After the conversation, Simple AI automatically transcribes the call, extracts key insights such as sentiment and outcome, and triggers follow-up workflows like updating CRM records, sending emails, or escalating issues to human agents. Its enterprise-grade security features—SOC 2 Type II certification, HIPAA compliance, and AES-256 encryption—ensure that all data remains secure and private. With these capabilities, businesses can use Simple AI for customer support, lead qualification, outbound sales, surveys, appointment reminders, or debt collection—essentially any repetitive voice-based process. The platform provides 24/7 availability, smooth human handoffs for complex cases, and scalable automation that reduces operational costs while maintaining a high-quality, consistent customer experience."}'''

        FORMAT = '''
        ```json
        [
            {
                "agent": "string",
                "prompt": "string",
                "label": "string"
            },
            {
                "agent": "string",
                "prompt": "string",
                "label": "string",
            }
        ]
        ```
        '''
        LLM_PROMPT = f"Break {PROMPT} into smaller tasks to be handled by {AGENTS} and provide tuples (agent, prompt, label) for each task where agent is which agent to complete that task and prompt is what to prompt that agent and label is a label for the particular task. The information produced by one agent can go into another. Format the output into a JSON object that contains fields for the specific agent and its prompt Format it like this: {FORMAT}. Do not include anything else."

        agent_dict = json.loads(AGENTS)
        client = InferenceClient("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

        messages = [
            {"role": "user", "content": LLM_PROMPT}
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
        )

        # print(response.choices[0].message["content"])

        output = response.choices[0].message["content"]

        try:
            cleaned_output = output.split('json', 1)[1]
            cleaned_output = cleaned_output.replace("```", "")
            data = json.loads(cleaned_output)
            n = len(data)
            for i in range(n):
                prompt = data[i]
                agent_name, agent_prompt = prompt['agent'], prompt['prompt']
                model_desc = agent_dict[agent_name]
                
                yield f"data: {json.dumps({"progress": f"{i+1}/{n}"})}\n\n"
                
                client = InferenceClient("Qwen/Qwen3-235B-A22B")
                
                prompt = f"You are a/an {agent_name} agent. You have the following capabilities: {model_desc}. Simulate the following task and come up with what the sample output should be: {agent_prompt}"
                messages = [
                    {"role": "user", "content": prompt}
                ]

                agent_response = client.chat_completion(
                    messages,
                    max_tokens=2000,
                    temperature=0.7,
                )
                data[i]['output'] = agent_response.choices[0].message["content"]
        
            yield f"data: {json.dumps({"progress": "done", "result": data})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({"error": str(e)})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# @app.post("/process")
# def process(data: Input):

#     print("running")
#     PROMPT = data.text
#     AGENTS = '''{"LinkedIn": "The AI-Assisted Search feature in LinkedIn Recruiter leverages Generative AI (GAI) to streamline and enhance the talent discovery process. Instead of relying solely on traditional keyword filters and Boolean strings, recruiters can now describe in natural language what kind of candidate they’re looking for—such as specific skills, experience levels, or job histories—and the AI automatically interprets and generates a targeted search query. This allows LinkedIn Recruiter to deliver a refined list of candidates who closely match the desired profile, even if their résumés use different terminology. The system learns from recruiter interactions and feedback to continuously improve search precision over time. In addition, it provides context-aware recommendations, highlights transferable skills, and surfaces diverse talent pools that might have been missed through conventional searches. Overall, AI-Assisted Search helps recruiters save time, reduce manual search complexity, and discover high-quality candidates more efficiently while maintaining a human-centered approach to final selection and outreach.",
#         "Gemini": "Content creation tools enable users to generate and refine various types of materials efficiently. For writing, AI can produce first drafts for emails and documents, help brainstorm ideas, and refine content for clarity and tone. Presentations can be transformed from simple ideas into polished slides with AI-generated visuals. In spreadsheets, AI assists with organizing and processing data, creating templates, and automatically filling in missing values. It can even generate video content from text prompts or images, creating cinematic scenes or short clips. In information and data analysis, AI can automatically summarize long documents, meetings, and message threads, analyze raw data to provide insights and formatting suggestions, and use research tools like NotebookLM to extract instant insights and summaries from uploaded sources. It can also perform intelligent searches across platforms like Google Drive by understanding user intent and context. For collaboration and communication, AI can generate meeting agendas, transcribe conversations in real time, and summarize discussions with clear action items. It provides real-time translation to support international teams and automates workflows across Google products—such as transcribing meeting notes and turning them into checklists. Additional capabilities include enhanced communication features that suggest better phrasing, flag misunderstandings, and prioritize important messages, as well as customer service support that helps build customer profiles and draft responses efficiently.",
#         "SimpleAI(S24)": "Simple AI is an advanced conversational platform designed to automate and enhance voice-based customer interactions using artificial intelligence. It functions as a virtual phone agent that can make and receive calls, understand natural language, and respond intelligently based on real-time context and company data. When a call is triggered—either through a CRM system, an API, or automated scheduling—Simple AI pulls relevant customer information, such as name, account details, and previous interactions, to personalize the conversation. The AI then engages the caller naturally, handling inquiries, objections, or routine transactions with human-like dialogue. It can access your company’s knowledge base in real time to provide accurate information, support multiple languages, and even navigate complex IVR menus. After the conversation, Simple AI automatically transcribes the call, extracts key insights such as sentiment and outcome, and triggers follow-up workflows like updating CRM records, sending emails, or escalating issues to human agents. Its enterprise-grade security features—SOC 2 Type II certification, HIPAA compliance, and AES-256 encryption—ensure that all data remains secure and private. With these capabilities, businesses can use Simple AI for customer support, lead qualification, outbound sales, surveys, appointment reminders, or debt collection—essentially any repetitive voice-based process. The platform provides 24/7 availability, smooth human handoffs for complex cases, and scalable automation that reduces operational costs while maintaining a high-quality, consistent customer experience."}'''

#     FORMAT = '''
#     ```json
#     [
#         {
#             "agent": "string",
#             "prompt": "string",
#             "label": "string"
#         },
#         {
#             "agent": "string",
#             "prompt": "string",
#             "label": "string",
#         }
#     ]
#     ```
#     '''
#     LLM_PROMPT = f"Break {PROMPT} into smaller tasks to be handled by {AGENTS} and provide tuples (agent, prompt, label) for each task where agent is which agent to complete that task and prompt is what to prompt that agent and label is a label for the particular task. The information produced by one agent can go into another. Format the output into a JSON object that contains fields for the specific agent and its prompt Format it like this: {FORMAT}. Do not include anything else."

#     agent_dict = json.loads(AGENTS)
#     client = InferenceClient("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

#     messages = [
#         {"role": "user", "content": LLM_PROMPT}
#     ]

#     response = client.chat_completion(
#         messages=messages,
#         max_tokens=2000,
#         temperature=0.7,
#     )

#     # print(response.choices[0].message["content"])

#     output = response.choices[0].message["content"]

#     try:
#         cleaned_output = output.split('json', 1)[1]
#         cleaned_output = cleaned_output.replace("```", "")
#         data = json.loads(cleaned_output)
#         n = len(data)
#         for i in range(n):
#             prompt = data[i]
#             agent_name, agent_prompt = prompt['agent'], prompt['prompt']
#             model_desc = agent_dict[agent_name]
#             print(f'{i+1}/{n}')
#             client = InferenceClient("Qwen/Qwen3-235B-A22B")
            
#             prompt = f"You are a/an {agent_name} agent. You have the following capabilities: {model_desc}. Simulate the following task and come up with what the sample output should be: {agent_prompt}"
#             messages = [
#                 {"role": "user", "content": prompt}
#             ]

#             agent_response = client.chat_completion(
#                 messages,
#                 max_tokens=2000,
#                 temperature=0.7,
#             )
#             data[i]['output'] = agent_response.choices[0].message["content"]
    
#         return {"result": data}
#     except Exception as e:
#         return {"error": str(e)}


