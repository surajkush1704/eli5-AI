# ============================================================
# ELI5 AI — Complex things. Simple words.
# Built with Streamlit + Google Gemini
# Author: Suraj Kumar
# ============================================================

import streamlit as st
import google.generativeai as genai
import json
import time
from dotenv import load_dotenv
import os

# ─── Load API Key ──────────────────────────────────────────
load_dotenv()

def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="ELI5 AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0A0A0F !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif;
  }
  #MainMenu, footer, header, [data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
  }
  .hero-title {
    font-family: 'Cinzel', serif;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #D4AF37, #F5D060, #D4AF37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 0.9rem;
    color: #777;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    color: #A78BFA;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
  }
  .result-card {
    background: linear-gradient(145deg, rgba(124,58,237,0.08), rgba(19,19,26,0.95));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
  }
  .result-emoji {
    font-size: 3.5rem;
    text-align: center;
    display: block;
    margin-bottom: 1rem;
  }
  .result-text {
    font-size: 1.1rem;
    line-height: 1.9;
    color: #D0D0E0;
    font-weight: 300;
  }
  .pill {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
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
  .limit-card {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
  }
  .countdown {
    font-family: 'Cinzel', serif;
    font-size: 2.8rem;
    color: #D4AF37;
    letter-spacing: 8px;
    margin: 1rem 0 0.3rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
  }
  .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
  }
  .stTextArea textarea {
    background: #1C1C27 !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    color: #E8E8F0 !important;
    font-size: 1rem !important;
  }
  .stTextArea textarea::placeholder { color: #555 !important; }
  .stTextArea textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
  }
  .stSelectbox > div > div {
    background: #1C1C27 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E0E0F0 !important;
  }
  hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0
if "reset_at" not in st.session_state:
    st.session_state.reset_at = time.time() + 3600
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "prefill_topic" not in st.session_state:
    st.session_state.prefill_topic = ""


# ─── Reset If Window Expired ───────────────────────────────
def maybe_reset():
    if time.time() >= st.session_state.reset_at:
        st.session_state.usage_count = 0
        st.session_state.reset_at = time.time() + 3600

maybe_reset()


# ─── Prompt Builder ────────────────────────────────────────
def build_prompt(topic, level, style):
    levels = {
        "👶 Kid":   "Explain to a 5-year-old. Very simple words. Max 3 short sentences. Start with something surprising.",
        "🧒 Teen":  "Explain to a 13-year-old. Simple but slightly technical. 4-5 sentences with a real example.",
        "🧑 Adult": "Explain for a smart adult. 6-7 clear sentences. Include why this matters.",
    }
    styles = {
        "📖 Simple Story": "Use a short story to explain.",
        "🎮 Video Game":   "Use gaming references and mechanics as analogies.",
        "🍕 Cooking":      "Use cooking and food as analogies.",
        "🎬 Movie Scene":  "Use movie scenes and cinema references.",
        "🌍 Real Life":    "Use everyday real-life examples.",
        "😂 Comedy":       "Explain in a funny, witty comedy style.",
    }
    return f"""You are ELI5 AI. {levels[level]} {styles[style]}

Topic: {topic}

Reply in EXACTLY this format:

EMOJI: [one emoji]

EXPLANATION:
[explanation here]

QUESTIONS:
{{"questions": ["question 1", "question 2", "question 3"]}}"""


# ─── Parse Response ────────────────────────────────────────
def parse_response(raw):
    emoji, explanation, questions = "🧠", raw, []
    try:
        for line in raw.splitlines():
            if line.strip().startswith("EMOJI:"):
                emoji = line.split("EMOJI:")[-1].strip()
                break
        if "EXPLANATION:" in raw:
            part = raw.split("EXPLANATION:")[-1]
            explanation = part.split("QUESTIONS:")[0].strip() if "QUESTIONS:" in part else part.strip()
        if "QUESTIONS:" in raw:
            q_part = raw.split("QUESTIONS:")[-1].strip()
            s, e = q_part.find("{"), q_part.rfind("}") + 1
            if s != -1 and e > s:
                questions = json.loads(q_part[s:e]).get("questions", [])
    except Exception:
        pass
    return emoji, explanation, questions


# ─── Limit Screen ──────────────────────────────────────────
def show_limit_screen():
    facts = [
        "Sharks are older than trees 🦈",
        "Honey never spoils 🍯",
        "The first computer bug was a real moth 🦋",
        "A day on Venus is longer than its year ☀️",
        "More people visited space than the ocean floor 🚀",
        "The word 'robot' means 'forced labor' in Czech 🤖",
    ]
    remaining = max(0, int(st.session_state.reset_at - time.time()))
    if remaining <= 0:
        st.session_state.usage_count = 0
        st.session_state.reset_at = time.time() + 3600
        st.rerun()
        return
    hrs = remaining // 3600
    mins = (remaining % 3600) // 60
    secs = remaining % 60
    st.markdown(f"""
    <div class="limit-card">
      <div style="font-size:3.5rem;">⏳</div>
      <div style="font-family:'Cinzel',serif; font-size:1.5rem; color:#E0E0F0; margin:0.5rem 0;">
        All 5 explanations used!
      </div>
      <div style="color:#777; margin-bottom:1rem;">Your brain needs a rest too 😄</div>
      <div class="countdown">{hrs:02d} : {mins:02d} : {secs:02d}</div>
      <div style="font-size:0.65rem; color:#555; letter-spacing:6px; text-transform:uppercase; margin-bottom:1.5rem;">
        HRS &nbsp; MIN &nbsp; SEC
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"💡 Did you know? {facts[int(time.time() / 5) % len(facts)]}")
    if st.button("🔄 Check If Reset", use_container_width=True):
        maybe_reset()
        st.rerun()


# ════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════

# ── HERO ──────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem;">
  <div class="hero-title">🧠 ELI5 AI</div>
  <div class="hero-sub">Complex things. Simple words.</div>
  <span class="hero-badge">⚡ 5 free explanations per hour</span>
</div>
""", unsafe_allow_html=True)

# ── USAGE BAR ─────────────────────────────────────────────
count = st.session_state.usage_count
filled = "▓" * count + "░" * (5 - count)
bar_color = "#4ADE80" if count <= 2 else ("#FBBF24" if count <= 4 else "#F87171")
st.markdown(f"""
<div style="text-align:center; margin-bottom:0.5rem;">
  <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);
    border-radius:50px; padding:0.3rem 1rem; font-size:0.85rem; color:#999;">
    {count} of 5 used &nbsp;
    <span style="color:{bar_color}; letter-spacing:2px;">{filled}</span>
  </span>
</div>
""", unsafe_allow_html=True)
st.progress(count / 5)
st.markdown("<br>", unsafe_allow_html=True)

# ── LIMIT CHECK ───────────────────────────────────────────
if st.session_state.usage_count >= 5:
    show_limit_screen()
    st.stop()

# ── API KEY CHECK ─────────────────────────────────────────
api_key = get_api_key()
if not api_key:
    st.error("⚠️ No Gemini API key found!")
    st.info("Add GEMINI_API_KEY to your .env file and restart the app.")
    st.code('GEMINI_API_KEY=your_key_here', language="bash")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"❌ Could not connect to Gemini: {e}")
    st.stop()

# ── TOPIC INPUT ───────────────────────────────────────────
prefill = st.session_state.prefill_topic
st.session_state.prefill_topic = ""

st.markdown('<p style="font-size:0.7rem; letter-spacing:2.5px; text-transform:uppercase; color:#555; font-weight:600;">WHAT\'S CONFUSING YOU?</p>', unsafe_allow_html=True)
topic = st.text_area(
    label="topic",
    label_visibility="collapsed",
    placeholder="Try: quantum physics, how planes fly, what is blockchain, why is the sky blue...",
    value=prefill,
    max_chars=200,
    height=100,
)
char_count = len(topic)
char_color = "#7C3AED" if char_count < 180 else "#F87171"
st.markdown(f'<div style="text-align:right; font-size:0.75rem; color:{char_color};">{char_count}/200</div>', unsafe_allow_html=True)

# ── SETTINGS ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="font-size:0.7rem; letter-spacing:2.5px; text-transform:uppercase; color:#555; font-weight:600;">WHO ARE YOU?</p>', unsafe_allow_html=True)
    level = st.radio("level", ["👶 Kid", "🧒 Teen", "🧑 Adult"],
                     horizontal=True, label_visibility="collapsed")
with col2:
    st.markdown('<p style="font-size:0.7rem; letter-spacing:2.5px; text-transform:uppercase; color:#555; font-weight:600;">EXPLAIN IT LIKE...</p>', unsafe_allow_html=True)
    style = st.selectbox("style",
                         ["📖 Simple Story", "🎮 Video Game", "🍕 Cooking",
                          "🎬 Movie Scene", "🌍 Real Life", "😂 Comedy"],
                         label_visibility="collapsed")

# ── EXPLAIN BUTTON ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
clicked = st.button("Explain It 🧠", use_container_width=True, type="primary")

# ── SHOW PREVIOUS RESULT ──────────────────────────────────
if st.session_state.last_result and not clicked:
    r = st.session_state.last_result
    st.markdown(f"""
    <div class="result-card">
      <span class="result-emoji">{r['emoji']}</span>
      <div class="result-text">{r['explanation']}</div>
      <div>
        <span class="pill pill-purple">{r['level']}</span>
        <span class="pill pill-gold">{r['style']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.code(r['explanation'], language=None)
    if r.get('questions'):
        st.markdown("---")
        st.markdown("**🤔 Curious about more?**")
        cols = st.columns(len(r['questions']))
        for i, (c, q) in enumerate(zip(cols, r['questions'])):
            with c:
                if st.button(q, key=f"pq_{i}", use_container_width=True):
                    st.session_state.prefill_topic = q
                    st.rerun()

# ── GENERATE ──────────────────────────────────────────────
if clicked:
    if not topic.strip():
        st.warning("⬆️ Type something above first! 🤔")
        st.stop()

    if st.session_state.usage_count >= 5:
        st.rerun()
        st.stop()

    with st.spinner("Making it simple... ✨"):
        try:
            prompt = build_prompt(topic.strip(), level, style)
            response = model.generate_content(prompt)
            raw = response.text
            emoji, explanation, questions = parse_response(raw)

            # ✅ Increment ONLY after success
            st.session_state.usage_count += 1
            if st.session_state.usage_count == 1:
                st.session_state.reset_at = time.time() + 3600

            # Save result
            st.session_state.last_result = {
                "emoji": emoji,
                "explanation": explanation,
                "questions": questions,
                "level": level,
                "style": style,
            }

            # Save to history
            st.session_state.history.append({
                "topic": topic.strip(),
                "emoji": emoji,
                "explanation": explanation,
                "level": level,
            })
            st.session_state.history = st.session_state.history[-3:]

            # Show result
            st.markdown(f"""
            <div class="result-card">
              <span class="result-emoji">{emoji}</span>
              <div class="result-text">{explanation}</div>
              <div>
                <span class="pill pill-purple">{level}</span>
                <span class="pill pill-gold">{style}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.code(explanation, language=None)

            if questions:
                st.markdown("---")
                st.markdown("**🤔 Curious about more?**")
                cols = st.columns(len(questions))
                for i, (c, q) in enumerate(zip(cols, questions)):
                    with c:
                        if st.button(q, key=f"q_{i}", use_container_width=True):
                            st.session_state.prefill_topic = q
                            st.rerun()

        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["quota", "429", "rate", "exhausted"]):
                st.error("⏳ Gemini API rate limit hit!")
                st.info("""
**This is a Gemini free tier limit — not our 5-search limit.**

Fix options:
1. **Wait 1 minute** and try again
2. **Get your own free API key** from [aistudio.google.com](https://aistudio.google.com/apikey) and add it to your .env file
                """)
            else:
                st.error(f"❌ Error: {str(e)}")

# ── HISTORY ───────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<p style="font-family:Cinzel,serif; font-size:1rem; color:#888; letter-spacing:1px;">📚 YOUR HISTORY</p>', unsafe_allow_html=True)
    for i, item in enumerate(reversed(st.session_state.history[-3:])):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
              border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.6rem;">
              <div style="font-weight:600; color:#C0C0D0;">{item['emoji']} {item['topic']}</div>
              <div style="font-size:0.82rem; color:#555; margin-top:0.2rem;">
                {item['explanation'][:80]}...
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if st.button("↩", key=f"h_{i}"):
                st.session_state.prefill_topic = item["topic"]
                st.rerun()

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.8rem; padding:1rem 0;">
  Made with ❤️ by 
  <a href="https://linkedin.com/in/surajkumar1704" style="color:#7C3AED; text-decoration:none;">Suraj Kumar</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/surajkush1704" style="color:#7C3AED; text-decoration:none;">GitHub</a>
  &nbsp;·&nbsp; Powered by Gemini AI 🤖
</div>
""", unsafe_allow_html=True)
