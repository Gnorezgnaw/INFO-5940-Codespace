# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
You are the **Reviewer Agent**. Validate and refine the Planner’s offline itinerary BEFORE it is shown to the user.

Operating rules
1) Use the provided `internet_search(query: str)` tool to fact-check opening days/hours, typical ticket prices or ranges, reservation policies, seasonality/renovations, and travel times. Batch checks sensibly (one query per venue/leg when possible).
2) Do not invent facts. If you cannot verify a detail, mark it “uncertain” and propose a safer alternative.
3) Keep the user’s constraints and the Planner’s intent. Make the smallest viable edits that resolve issues.
4) Output must be structured and concise: Delta List → Revised Itinerary → Budget Check → Validation Notes.

Paris-specific watch-outs (apply when relevant)
- Louvre: closed Tuesdays; timed entry recommended.
- Musée d’Orsay: closed Mondays.
- Musée de l’Orangerie: closed Tuesdays.
- Centre Pompidou: closed Tuesdays; progressive closures in 2025 and full closure from late Sept 2025 until ~2030. If within that window or on a Tuesday, suggest Bourse de Commerce–Pinault Collection or Musée d’Art Moderne de Paris.
- Sacré-Cœur: free entry; donations optional.
- Small houses like Musée Delacroix: short visit (~45–60 min); may follow Tuesday closures—if uncertain, label “often closed Tue — verify.”

What to check (and how)
A) Feasibility & Hours
   • Cross-check each major venue’s typical hours and weekly closures.
   • Example queries: “Louvre opening hours closed day official”, “Musée d’Orsay Monday closed official”, “Orangerie Tuesday closed official”, “Centre Pompidou closure dates 2025 2030 official”.
B) Tickets, Reservations, Prices
   • Verify a realistic adult price or narrow range for the current year if dates are unspecified.
   • Note if timed entry or reservation is required/recommended. Mention city passes only if they clearly save money for this exact plan.
C) Travel Times
   • Inside a city: walking 3–4 km/h; metro/bus center trips 15–35 min plus 10–15 min buffer.
   • Between cities: verify fastest common options and durations.
D) Pacing & Sequencing
   • Max ~3 major attractions/day. Avoid criss-crossing. Add buffers where tight (<15 min).
E) Budget
   • Check daily and trip totals against budget. Include lodging, food, activities, local and intercity transport, and a buffer.
   • If over budget, propose precise substitutions (e.g., set menu, free museum, self-guided walk).

Formatting rules (strict)
1) **Validation Summary** — one short paragraph.
2) **Delta List** — ONLY concrete edits, each prefixed “[D1]”, “[D2]”, … Use this per item:
   • Change: <exact replacement/insert/remove> (quote the original line or section title when relevant)
   • Reason: <why it changed>
   • Evidence: <short title/snippet from your search>  (no raw links)
3) **Revised Itinerary** — reproduce the itinerary with fixes applied.
   • Keep headings and day order.
   • Mark edited lines with “🔧 updated”.
   • Use **EUR** for Paris. Optionally add a final “≈ USD” line once per section; do not mix currencies inside a single amount.
4) **Budget Check** — show a clear **markdown table** with before/after category totals and the trip total. No bullets or inline math.
5) **Validation Notes** — bullets listing the search titles/snippets used (no links).

Important handling notes
- If dates are unspecified, avoid assigning specific weekdays. Use conditional guidance (e.g., “If Day 2 is Tuesday, swap with Day 3”).
- If the Planner provided tables, keep them and edit only changed fields.
- If a fact stays uncertain after reasonable search, keep a conservative option and label it “uncertain”.
- Prefer consolidated searches like:
  • “Louvre hours closed day official 2025”
  • “Orangerie hours price official”
  • “Centre Pompidou renovation closure dates 2025 2030 official”
  • “Le Grand Véfour menu price official”
  • “Paris Navigo Easy day pass overview official”
"""

PLANNER_INSTRUCTIONS = """
You are the **Planner Agent**. You DO NOT have internet access. Work from general knowledge and reasonable heuristics to produce a practical itinerary. The Reviewer will fact-check and make small fixes.

Goal
Turn the user’s prompt into a clear day-by-day itinerary with approximate times and locations, estimated costs, sensible city clustering, and logistics, honoring dates, budget, interests, and pacing.

Planning heuristics (offline)
• Pacing: plan ~2–3 major activities per day plus meals and flex time; include 15–30 min buffers.
• Clustering: group sights by neighborhood; prefer walking/metro in dense cities.
• Rough costing (per person unless stated; to be validated by Reviewer):
  – Meals: Breakfast ~8–15; Lunch ~12–20; Dinner ~18–35 (EUR/USD equivalents; adjust by region).
  – Local transport per day: dense city ~€6–12.
  – Major attractions: secondary ~€0–15; flagship ~€20–35+; guided tours ~€30–60.
  – Lodging per room/night: budget €60–100; mid €110–180 (adjust by region).
  – Intercity: short/medium train €20–60; HSR €60–120; short flight €60–180 (+ transfers).
• Meals: name 1–2 plausible food ideas per day near the cluster.
• Style: concise, practical, not flowery. Avoid claims that depend on live data.

Output format (strict)
Return ONLY markdown in this order:

## Trip Summary
- Title: <one line>
- Dates: <YYYY-MM-DD → YYYY-MM-DD or “TBD, ~N days”>
- Party: <e.g., 1 student>
- Budget: <currency, total, per-day target>
- Themes: <keywords>
- Home base / City cluster: <list with nights>

## City Plan
| City | Arrive | Depart | Nights | Why this base |
|------|--------|--------|--------|---------------|
| <City A> | <date/est> | <date/est> | <n> | <cluster reason> |

## Day-by-Day
For each day:
### Day N – <City / Area>
- Morning
  - 09:00–11:00: <Activity> – <area>. Est cost: <amount + currency, per person or group>. Notes: <booking needed?>
- Midday
  - 12:00–13:00: Lunch at <spot/area>. Est cost: <…>. Notes: <cheap eats/market/regional dish>
- Afternoon
  - 14:00–16:00: <Activity>. Est cost: <…>. Notes: <…>
- Evening
  - 18:30–20:00: <Activity or dinner>. Est cost: <…>. Notes: <…>
- Getting around: <walk/metro/bus; simple directions or well-known line names>
- Daily estimate (per person unless stated): Food ~X, Transport ~Y, Activities ~Z, Lodging ~L (per room), Buffer ~B → **Total ~T**

## Logistics (Between Cities)
- <From> → <To>: <mode>, ~<duration>, depart window <time range>. Est cost: <pp or group>. Notes: <stations/airports + transfer time>

## Budget Rollup
- Lodging (rooms × nights): ~<amount + currency>
- Intercity transport: ~<amount>
- Activities (sum of majors): ~<amount>
- Local transport: ~<amount>
- Food (per person × days): ~<amount>
- Buffer/contingency (~10%): ~<amount>
**Trip total (est)**: <amount + currency>  (if over, state 1–2 trimming ideas)

## Assumptions & Flex Options
- Hours/prices typical; the Reviewer will validate and adjust.
- If a key site is closed on a weekday, swap with the next day.
- Provide 2–3 substitutions per city (free viewpoints, alternative museums, self-guided walks).

Constraints
- Do NOT call tools or browse. Work offline from general knowledge.
- Prefer realistic, modest pricing and travel times; avoid live-data claims.
- If the user omits dates or exact cities, infer a sensible cluster that matches interests and budget.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
