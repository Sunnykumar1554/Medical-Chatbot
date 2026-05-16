from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from functools import wraps
from src.helper import download_hugging_face_embeddings
from user_db import UserDB
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from src.prompt import *
import os
import sys
import base64
import re
from openai import OpenAI


# Fix Windows console encoding (cp1252 can't handle unicode from LLM)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Max upload size: 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

load_dotenv()

PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY")
NVIDIA_API_KEY    = os.environ.get("NVIDIA_API_KEY", "")
LLAMA_MODEL       = os.environ.get("LLAMA_MODEL", "llama3")
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["LLAMA_MODEL"]      = LLAMA_MODEL

# ── Embeddings & Vector Store (medical knowledge — untouched namespace) ──────
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch  = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.7}
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

chatModel = ChatOllama(
    model=LLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.1")),
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain             = create_retrieval_chain(retriever, question_answer_chain)

# ── User DB (separate Pinecone namespaces — medical data never touched) ──────
user_db = UserDB(pinecone_api_key=PINECONE_API_KEY, index_name=index_name)

# ── Google OAuth ─────────────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ─────────────────────────── Helpers ─────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def _user_context_prefix(user_meta: dict) -> str:
    """Prepend age/gender to every query so the LLM skips asking for it."""
    age    = int(user_meta.get("age", 0))
    gender = user_meta.get("gender", "")
    parts  = []
    if age:    parts.append(f"age {age}")
    if gender: parts.append(f"gender {gender}")
    return f"[Patient profile: {', '.join(parts)}] " if parts else ""


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def markdown_to_html(text: str) -> str:
    """Convert GPT-4o markdown response to HTML for the chat bubble."""
    # Section headers (##)
    text = re.sub(r'^## (.+)$', r'<h4 class="rx-section-title">\1</h4>', text, flags=re.MULTILINE)
    # Medicine name headers (###)
    text = re.sub(r'^### (.+)$', r'<h5 class="rx-med-name">\1</h5>', text, flags=re.MULTILINE)
    # Bold text
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Bullet list items
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Wrap consecutive <li> runs in <ul>
    text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: '<ul class="rx-list">' + m.group(0) + '</ul>', text, flags=re.DOTALL)
    # Line breaks
    text = text.replace('\n', '<br>')
    # Clean up extra <br> adjacent to block elements
    text = re.sub(r'<br>(<(?:h4|h5|ul|li))', r'\1', text)
    text = re.sub(r'(</(?:h4|h5|ul|li)>)<br>', r'\1', text)
    # Horizontal rule
    text = text.replace('---', '<hr class="rx-divider">')
    return text


# ─────────────────────────── Auth Routes ─────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    user = user_db.login(username, password)
    if not user:
        return render_template("login.html", error="Invalid username or password.")

    session["user_id"]  = user["user_id"]
    session["username"] = user["username"]
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    email    = request.form.get("email", "").strip()
    age_raw  = request.form.get("age", "").strip()
    gender   = request.form.get("gender", "").strip()

    if not username or not password:
        return render_template("register.html", error="Username and password are required.")

    age    = int(age_raw) if age_raw.isdigit() else None
    result = user_db.register(username=username, password=password,
                              email=email, age=age, gender=gender)
    if not result["success"]:
        return render_template("register.html", error=result["error"])

    session["user_id"]  = result["user_id"]
    session["username"] = username
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────── Google OAuth Routes ─────────────────────────────

@app.route("/auth/google")
def google_login():
    """Redirect user to Google consent screen."""
    redirect_uri = "http://localhost:5000/auth/google/callback"
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    """Handle Google OAuth callback — login or auto-register."""
    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            user_info = google.userinfo()
    except Exception as e:
        print(f"[GOOGLE AUTH ERROR] {e}")
        return render_template("login.html", error="Google sign-in failed. Please try again.")

    email = user_info.get("email", "").strip().lower()
    name  = user_info.get("name", "")

    if not email:
        return render_template("login.html", error="Could not retrieve email from Google.")

    # Find existing user or create new one
    result = user_db.find_or_create_google_user(email=email, name=name)

    session["user_id"]  = result["user_id"]
    session["username"] = result["username"]
    return redirect(url_for("index"))



@app.route("/")
@login_required
def index():
    user = user_db.get_user_by_id(session["user_id"])
    return render_template("chat.html", user=user)


@app.route("/profile", methods=["POST"])
@login_required
def update_profile():
    age_raw = request.form.get("age", "").strip()
    gender  = request.form.get("gender", "").strip()
    age     = int(age_raw) if age_raw.isdigit() else None
    user_db.update_profile(
        user_id=session["user_id"],
        age=age,
        gender=gender if gender else None,
    )
    return redirect(url_for("index"))


@app.route("/history")
@login_required
def history():
    msgs = user_db.get_history(session["user_id"], limit=100)
    return jsonify(msgs)


@app.route("/get", methods=["GET", "POST"])
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message.", 400

    user       = user_db.get_user_by_id(session["user_id"])
    prefix     = _user_context_prefix(user) if user else ""
    full_input = prefix + msg

    print(f"[{session.get('username')}] {full_input}")
    response = rag_chain.invoke({"input": full_input})
    answer   = response["answer"]
    print("Response:", answer)

    # Persist conversation
    user_db.save_message(session["user_id"], role="user", content=msg)
    user_db.save_message(session["user_id"], role="bot",  content=answer)

    return str(answer)


@app.route("/delete_conversation", methods=["POST"])
@login_required
def delete_conversation():
    """Delete a specific conversation pair (user msg + bot reply) by timestamps."""
    data = request.get_json()
    timestamps = data.get("timestamps", [])
    if not timestamps:
        return jsonify({"success": False, "error": "No timestamps provided"}), 400

    # Convert to floats
    ts_list = [float(t) for t in timestamps]
    result = user_db.delete_messages(session["user_id"], ts_list)
    return jsonify({"success": result})


@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    """Delete all chat history for the current user."""
    result = user_db.clear_history(session["user_id"])
    return jsonify({"success": result})


@app.route("/analyze-prescription", methods=["POST"])
@login_required
def analyze_prescription():
    """
    Accept an uploaded prescription image, send it to NVIDIA Nemotron VL
    (llama-3.1-nemotron-nano-vl-8b-v1) via the OpenAI-compatible NIM API,
    and return an HTML-formatted analysis for the chat bubble.
    """
    if 'prescription' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['prescription']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, WEBP, or GIF image."}), 400

    if not NVIDIA_API_KEY:
        return jsonify({"error": "NVIDIA API key not configured. Add NVIDIA_API_KEY to your .env file. Get one at https://build.nvidia.com"}), 500

    # Read the image bytes and encode to base64
    file_bytes = file.read()
    ext = file.filename.rsplit('.', 1)[1].lower()
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
                'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp'}
    mime_type = mime_map.get(ext, 'image/jpeg')
    image_b64 = base64.b64encode(file_bytes).decode('utf-8')

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
        )

        user_prompt = PRESCRIPTION_SYSTEM_PROMPT + "\n\nPlease analyze this prescription image and provide a full breakdown."

        print("[Prescription] Sending to NVIDIA Nemotron VL...")
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=2048,
        )

        raw_text = response.choices[0].message.content

        if not raw_text:
            return jsonify({"error": "Model returned an empty response. Try again."}), 500

        print(f"[Prescription] Analysis complete ({len(raw_text)} chars)")
        html_response = markdown_to_html(raw_text)

        # Persist to chat history so it appears in past conversations
        user_label = "📋 Prescription uploaded"
        if request.form.get("note", "").strip():
            user_label += f" — {request.form['note'].strip()}"
        user_db.save_message(session["user_id"], role="user",
                             content=f"[PRESCRIPTION] {user_label}")
        user_db.save_message(session["user_id"], role="bot",
                             content=f"[PRESCRIPTION_ANALYSIS]\n{raw_text}")

        return jsonify({"analysis": html_response})

    except Exception as e:
        print(f"[Prescription Analysis Error] {e}")
        err_str = str(e)
        if "401" in err_str or "403" in err_str or "API_KEY" in err_str.upper():
            return jsonify({"error": "Invalid NVIDIA API key. Check NVIDIA_API_KEY in your .env file."}), 500
        if "429" in err_str:
            return jsonify({"error": "Rate limit reached. Wait a moment and try again."}), 500
        return jsonify({"error": f"Analysis failed: {err_str}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.chat_models import ChatOllama
# # from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate       
# from dotenv import load_dotenv
# from src.prompt import *
# import os


# app = Flask(__name__)


# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3")
# OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# # OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["LLAMA_MODEL"] = LLAMA_MODEL
# # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# embeddings = download_hugging_face_embeddings()

# index_name = "medical-chatbot" 
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )




# retriever = docsearch.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.7}
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# chatModel = ChatOllama(
#     model=LLAMA_MODEL,
#     base_url=OLLAMA_BASE_URL,
#     temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.1")),
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# @app.route("/")
# def index():
#     return render_template('chat.html')



# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])



# if __name__ == '__main__':
#     # choose a non-privileged default port (5000) with env override
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)