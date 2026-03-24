# ============================================================
# ELI5 AI — Complex things. Simple words.
# Built with Streamlit + Google Gemini
# Author: Suraj Kumar
# ============================================================

import streamlit as st
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# ─── Load environment variables ────────────────────────────
load_dotenv()
_DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")


def get_model(api_key: str):
    """Return a Gemini model client for the given API key."""
    try:
        from google import genai as _genai  # new SDK
        client = _genai.Client(api_key=api_key)
        return client, "new"
    except ImportError:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        return genai_old.GenerativeModel("gemini-1.5-flash"), "old"


def call_model(client_or_model, sdk_type: str, prompt: str) -> str:
    """Call the model and return text, trying multiple models as fallback."""
    if sdk_type == "new":
        # Try models in order of preference
        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
            try:
                response = client_or_model.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                err = str(e).lower()
                # Only fall through on rate/quota errors, not auth/404
                is_rate = any(k in err for k in ["429", "quota", "resource_exhausted", "too many", "rate"])
                if is_rate:
                    continue  # try next model
                raise  # re-raise non-rate errors immediately
        raise Exception("All models are rate-limited. Please wait and try again.")
    else:
        response = client_or_model.generate_content(prompt)
        return response.text

# ─── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title="ELI5 AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS — Dark Premium Theme ───────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Root dark background ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A0F !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif;
  }

  [data-testid="stMain"] {
    background: #0A0A0F !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  [data-testid="stToolbar"] {display: none;}

  /* ── Hero header ── */
  .hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #D4AF37 0%, #F5D060 50%, #D4AF37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    text-shadow: none;
    letter-spacing: 2px;
  }

  .hero-glow {
    text-align: center;
    position: relative;
    padding: 1.5rem 0 0.5rem;
  }

  .hero-glow::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 300px;
    height: 150px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.3) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
  }

  .hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #888;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  .hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    color: #A78BFA;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.5px;
  }

  /* ── Glassmorphism card ── */
  .glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
  }

  /* ── Result card ── */
  .result-card {
    background: linear-gradient(145deg, rgba(124,58,237,0.08), rgba(19,19,26,0.95));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(124,58,237,0.15), 0 0 0 1px rgba(255,255,255,0.03);
  }

  .result-emoji {
    font-size: 4rem;
    text-align: center;
    display: block;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 12px rgba(212,175,55,0.5));
  }

  .result-text {
    font-size: 1.15rem;
    line-height: 1.8;
    color: #D0D0E0;
    font-family: 'Inter', sans-serif;
    font-weight: 300;
  }

  /* ── Pill badges ── */
  .pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.4rem;
  }

  .pill-purple {
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #A78BFA;
  }

  .pill-gold {
    background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.35);
    color: #D4AF37;
  }

  /* ── Usage counter ── */
  .usage-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-size: 0.85rem;
    color: #999;
    font-family: 'Inter', sans-serif;
    display: inline-block;
  }

  /* ── History card ── */
  .history-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
  }

  .history-card:hover {
    border-color: rgba(124,58,237,0.3);
    background: rgba(124,58,237,0.05);
  }

  .history-topic {
    font-weight: 600;
    color: #C0C0D0;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
  }

  .history-snippet {
    font-size: 0.82rem;
    color: #666;
    line-height: 1.5;
  }

  /* ── Limit screen ── */
  .limit-card {
    background: linear-gradient(145deg, rgba(239,68,68,0.08), rgba(13,13,20,0.95));
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
  }

  .countdown-display {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    color: #D4AF37;
    text-align: center;
    letter-spacing: 8px;
    margin: 1rem 0 0.5rem;
  }

  .countdown-label {
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 6px;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.7rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #555;
    font-weight: 600;
    margin-bottom: 0.4rem;
    font-family: 'Inter', sans-serif;
  }

  /* ── Buttons override ── */
  .stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
  }

  .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Text area ── */
  .stTextArea textarea {
    background: #F8F8FF !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    color: #111111 !important;
    caret-color: #7C3AED !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
  }

  .stTextArea textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    background: #FFFFFF !important;
  }

  .stTextArea textarea::placeholder {
    color: #999 !important;
  }

  /* ── Select / Radio ── */
  .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E0E0F0 !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div {
    border-radius: 50px !important;
  }

  /* ── Spinner ── */
  .stSpinner > div {
    border-color: #7C3AED !important;
  }

  /* ── Divider ── */
  hr {
    border-color: rgba(255,255,255,0.06) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0D0D14 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
  }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ──────────────────────────
def init_session_state():
    """Initialize all session_state variables safely."""
    if "count" not in st.session_state:
        st.session_state.count = 0
    if "reset_time" not in st.session_state:
        st.session_state.reset_time = datetime.now() + timedelta(hours=1)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rerun_topic" not in st.session_state:
        st.session_state.rerun_topic = ""


# ─── Usage Limit Logic ─────────────────────────────────────
def check_usage_limit() -> bool:
    """Returns True if user has exceeded the hourly limit."""
    now = datetime.now()
    # Reset counter when window expires
    if now >= st.session_state.reset_time:
        st.session_state.count = 0
        st.session_state.reset_time = now + timedelta(hours=1)
    return st.session_state.count >= 5


# ─── Prompt Builder ────────────────────────────────────────
def build_prompt(topic: str, level: str, style: str) -> str:
    """Construct the Gemini prompt based on user selections."""
    levels = {
        "👶 Kid": (
            "Explain to a 5-year-old. "
            "Use VERY simple words only. "
            "Maximum 3 short sentences. "
            "Start with something surprising."
        ),
        "🧒 Teen": (
            "Explain to a 13-year-old. "
            "Simple but slightly technical. "
            "4-5 sentences. Include a real example."
        ),
        "🧑 Adult": (
            "Explain clearly for a smart adult. "
            "6-7 sentences. Include why this matters."
        ),
    }

    styles = {
        "📖 Simple Story": "Use a short story format to explain.",
        "🎮 Video Game": "Use gaming references and game mechanics as analogies.",
        "🍕 Cooking": "Use cooking, recipes, and food as analogies.",
        "🎬 Movie Scene": "Use movie scenes and cinema references as analogies.",
        "🌍 Real Life": "Use everyday objects and real daily-life examples.",
        "😂 Comedy": "Explain in a funny, witty, stand-up comedy style with jokes and humor. Make the user laugh while learning.",
    }

    return f"""You are ELI5 AI, a friendly genius who makes complex things simple.

{levels[level]}
{styles[style]}

Topic: {topic}

Respond in EXACTLY this format — do not deviate:

EMOJI: [one relevant emoji]

EXPLANATION:
[your explanation here]

QUESTIONS:
{{"questions": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]}}"""


# ─── Response Parser ───────────────────────────────────────
def parse_response(raw: str) -> tuple[str, str, list[str]]:
    """
    Parse the structured Gemini response.
    Returns: (emoji, explanation, questions_list)
    """
    emoji = "🧠"
    explanation = raw
    questions = []

    try:
        # Extract EMOJI
        for line in raw.splitlines():
            if line.strip().startswith("EMOJI:"):
                emoji = line.split("EMOJI:")[-1].strip()
                break

        # Extract EXPLANATION
        if "EXPLANATION:" in raw:
            expl_part = raw.split("EXPLANATION:")[-1]
            if "QUESTIONS:" in expl_part:
                explanation = expl_part.split("QUESTIONS:")[0].strip()
            else:
                explanation = expl_part.strip()

        # Extract QUESTIONS JSON
        if "QUESTIONS:" in raw:
            q_part = raw.split("QUESTIONS:")[-1].strip()
            # Find JSON object
            start = q_part.find("{")
            end = q_part.rfind("}") + 1
            if start != -1 and end > start:
                q_json = json.loads(q_part[start:end])
                questions = q_json.get("questions", [])

    except Exception:
        pass  # Graceful fallback — return what we have

    return emoji, explanation, questions


# ─── Show Usage Counter ────────────────────────────────────
def show_usage_counter():
    """Display the usage pill and progress bar."""
    count = st.session_state.count
    remaining = 5 - count

    # Build block string
    filled = "▓" * count
    empty = "░" * remaining
    bar_str = filled + empty

    # Color-coded label
    if count <= 2:
        color = "#4ADE80"  # green
    elif count <= 4:
        color = "#FBBF24"  # yellow
    else:
        color = "#F87171"  # red

    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:1rem;">
          <span class="usage-pill">
            {count} of 5 used &nbsp;
            <span style="color:{color}; letter-spacing:2px;">{bar_str}</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_val = count / 5
    st.progress(progress_val)


# ─── Limit Reached Screen ──────────────────────────────────
def show_limit_screen():
    """Full-page overlay when the user hits 5 explanations."""
    fun_facts = [
        "The word 'robot' comes from Czech meaning 'forced labor' 🤖",
        "Honey never spoils — 3000-year-old honey found in Egyptian tombs 🍯",
        "Sharks are older than trees 🦈",
        "The first computer bug was a real moth found in 1947 🦋",
        "A day on Venus is longer than its year ☀️",
        "More people have visited space than the bottom of the ocean 🚀",
    ]

    now = datetime.now()
    remaining = st.session_state.reset_time - now

    # Safety guard
    if remaining.total_seconds() <= 0:
        st.session_state.count = 0
        st.session_state.reset_time = now + timedelta(hours=1)
        st.rerun()

    total_secs = int(remaining.total_seconds())
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60

    st.markdown(
        f"""
        <div class="limit-card">
          <div style="font-size:4rem; margin-bottom:0.5rem;">⏳</div>
          <div style="font-family:'Cinzel',serif; font-size:1.6rem; color:#E0E0F0; margin-bottom:0.3rem;">
            You've used all 5 explanations!
          </div>
          <div style="color:#777; font-size:0.95rem;">Your brain needs a rest too 😄</div>
          <div class="countdown-display">{hrs:02d} : {mins:02d} : {secs:02d}</div>
          <div class="countdown-label">HRS &nbsp;&nbsp;&nbsp; MIN &nbsp;&nbsp;&nbsp; SEC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rotating fun fact
    fact_index = int(time.time() / 5) % len(fun_facts)
    st.info(f"💡 Did you know? {fun_facts[fact_index]}")

    st.link_button("Share ELI5 AI 🚀", "https://surajkush1704.github.io/eli5-ai",
                   use_container_width=True)

    # Auto-refresh every second for countdown
    time.sleep(1)
    st.rerun()


# ─── Recent History Section ────────────────────────────────
def show_history():
    """Render the last 3 items from the session history."""
    if not st.session_state.history:
        return

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Cinzel\',serif; font-size:1.1rem; color:#888; '
        'letter-spacing:1px; margin-bottom:0.8rem;">📚 YOUR HISTORY</div>',
        unsafe_allow_html=True,
    )

    for hist_idx, item in enumerate(reversed(st.session_state.history[-3:])):
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.markdown(
                f"""
                <div class="history-card">
                  <div class="history-topic">{item['emoji']} {item['topic']}</div>
                  <div class="history-snippet">{item['explanation'][:80]}...</div>
                  <div style="margin-top:0.4rem;">
                    <span class="pill pill-purple">{item['level']}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            if st.button("↩ View", key=f"hist_view_{hist_idx}"):
                st.session_state.rerun_topic = item["topic"]
                st.rerun()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 ELI5 AI")
    st.markdown("Made by **Suraj Kumar**")
    st.markdown("[🔗 LinkedIn](https://linkedin.com/in/surajkumar1704)")
    st.markdown("[🐙 GitHub](https://github.com/surajkush1704)")
    st.markdown("---")

    st.markdown("### 🔑 API Key")
    custom_key = st.text_input(
        "Gemini API Key",
        value=_DEFAULT_API_KEY,
        type="password",
        help="Paste a fresh key from aistudio.google.com/apikey if you hit rate limits",
        key="api_key_input",
    )
    st.markdown(
        "[🔗 Get a free key](https://aistudio.google.com/apikey)",
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.markdown("Powered by **Gemini AI** 🤖")
    st.markdown("*Complex things. Simple words.*")
    st.markdown("---")
    st.caption("v1.0 · Free to use · Made with ❤️")


# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════
def main():
    init_session_state()

    # ── Pick active API key (sidebar override > .env) ────────
    active_key = st.session_state.get("api_key_input", "").strip() or _DEFAULT_API_KEY
    if not active_key:
        st.error("⚠️ No API key found. Open the sidebar and paste your Gemini key.")
        st.stop()

    _client, _sdk = get_model(active_key)

    # ── HERO HEADER ──────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-glow">
          <div class="hero-title">🧠 ELI5 AI</div>
          <div class="hero-subtitle">Complex things. Simple words.</div>
          <div style="text-align:center;">
            <span class="hero-badge">⚡ 5 free explanations per hour</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── USAGE COUNTER ─────────────────────────────────────────
    show_usage_counter()

    # ── LIMIT CHECK ───────────────────────────────────────────
    if check_usage_limit():
        show_limit_screen()
        return  # Stop here — limit screen handles re-run

    # ── TOPIC INPUT ───────────────────────────────────────────
    # Pre-fill if user clicked a history / follow-up item
    default_topic = st.session_state.rerun_topic
    st.session_state.rerun_topic = ""  # Clear after reading

    st.markdown('<div class="section-label">WHAT\'S CONFUSING YOU?</div>', unsafe_allow_html=True)
    topic = st.text_area(
        label="topic_input",
        label_visibility="collapsed",
        placeholder="Try: quantum physics, how planes fly, what is blockchain, why is the sky blue...",
        value=default_topic,
        max_chars=200,
        height=100,
        key="topic_input",
    )

    # Character counter
    char_count = len(topic)
    char_color = "#7C3AED" if char_count < 180 else "#F87171"
    st.markdown(
        f'<div style="text-align:right; font-size:0.75rem; color:{char_color};">'
        f'{char_count}/200</div>',
        unsafe_allow_html=True,
    )

    # ── SETTINGS ROW ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">WHO ARE YOU?</div>', unsafe_allow_html=True)
        level = st.radio(
            label="level",
            label_visibility="collapsed",
            options=["👶 Kid", "🧒 Teen", "🧑 Adult"],
            horizontal=True,
            key="level_radio",
        )

    with col2:
        st.markdown('<div class="section-label">EXPLAIN IT LIKE...</div>', unsafe_allow_html=True)
        style = st.selectbox(
            label="style",
            label_visibility="collapsed",
            options=[
                "📖 Simple Story",
                "🎮 Video Game",
                "🍕 Cooking",
                "🎬 Movie Scene",
                "🌍 Real Life",
                "😂 Comedy",
            ],
            key="style_select",
        )

    # ── EXPLAIN BUTTON ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    clicked = st.button("Explain It 🧠", use_container_width=True, type="primary")

    # ── GENERATION ───────────────────────────────────────────
    if clicked:
        if not topic.strip():
            st.warning("⬆️ Type something in the box above first! 🤔")
            st.stop()

        # ── GENERATION with retry ────────────────────────────
        result_placeholder = st.empty()

        def call_gemini_with_retry(prompt_text: str, max_retries: int = 3):
            """Call Gemini with exponential backoff on rate limit errors."""
            for attempt in range(max_retries):
                try:
                    return call_model(_client, _sdk, prompt_text)
                except Exception as exc:
                    err = str(exc).lower()
                    is_rate_err = any(k in err for k in ["quota", "rate", "429", "resource_exhausted", "too many"])
                    if is_rate_err and attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        with result_placeholder.container():
                            st.warning(f"⏳ Gemini is busy — retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait)
                        result_placeholder.empty()
                        continue
                    raise exc  # re-raise on final attempt or non-rate errors

        try:
            with st.spinner("Making it simple... ✨"):
                prompt = build_prompt(topic.strip(), level, style)
                raw = call_gemini_with_retry(prompt)

            emoji, explanation, questions = parse_response(raw)

            # Increment usage counter on success
            st.session_state.count += 1

            # ── RESULT CARD ──────────────────────────────
            st.markdown(
                f"""
                <div class="result-card">
                  <span class="result-emoji">{emoji}</span>
                  <div class="result-text">{explanation}</div>
                  <br>
                  <div>
                    <span class="pill pill-purple">{level}</span>
                    <span class="pill pill-gold">{style}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── COPY BLOCK ───────────────────────────────
            st.code(explanation, language=None)

            # ── FOLLOW-UP QUESTIONS ──────────────────────
            if questions:
                st.markdown("---")
                st.markdown("**🤔 Curious about more?**")
                q_cols = st.columns(len(questions))
                for idx, (qcol, q) in enumerate(zip(q_cols, questions)):
                    with qcol:
                        if st.button(q, key=f"q_{idx}", use_container_width=True):
                            st.session_state.rerun_topic = q
                            st.rerun()

            # ── SAVE TO HISTORY ──────────────────────────
            st.session_state.history.append({
                "topic": topic.strip(),
                "emoji": emoji,
                "explanation": explanation,
                "level": level,
                "style": style,
            })
            st.session_state.history = st.session_state.history[-3:]

        except Exception as e:
            err_str = str(e).lower()
            is_rate = any(k in err_str for k in ["quota", "rate", "429", "resource_exhausted", "too many"])
            if is_rate:
                st.error(
                    "🚦 **Rate limit reached.** The free Gemini tier allows ~15 requests/minute. "
                    "Please wait a moment and try again, or open the **sidebar** to paste a fresh API key."
                )
                st.markdown("🔑 [Get a free Gemini API key](https://aistudio.google.com/apikey)")
            else:
                st.error("😅 **Something went wrong.** Please try again in a moment.")

    # ── HISTORY SECTION ───────────────────────────────────────
    show_history()


# ─── Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    main()
