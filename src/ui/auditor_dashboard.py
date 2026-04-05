"""
SmartForge — Auditor Dashboard (5-Tab)
========================================
Port 7861 · Audience: Insurance adjuster / compliance auditor
Role: AUDITOR — no vehicle_id filter → full case visibility

Tab 1  🗂️ Case Explorer      Search all cases, full detail on row-click
Tab 2  📋 Insurance Claims   All filed claims, approve/reject actions
Tab 3  🚨 Fraud Review       Suspicious cases, auditor decision write-back
Tab 4  👤 User Management    Per-vehicle aggregation + claim history
Tab 5  📊 Audit Logs         Agent trace, checkpoint dump, decision timeline
"""

import json as _json
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from src.config.settings import cfg
from src.db.mongo_client import (
    db_backend_info,
    db_find,
    db_get,
    db_mark_auditor,
    db_upsert,
)
from src.ui.helpers import build_stats_html
from src.ui.theme import fraud_badge, get_css_block, get_theme, score_badge

# ─────────────────────────────────────────────────────────────────────────────
# Shared DataFrames
# ─────────────────────────────────────────────────────────────────────────────

_CONF = {"confirmed", "gemini_golden_frame_confirmed", "gemini_confirmed"}


def _cases_to_df(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        fo = r.get("final_output") or {}
        fr = r.get("fraud_report") or {}
        rows.append({
            "Case ID":      (r.get("case_id", "") or "")[:20],
            "Vehicle ID":   r.get("vehicle_id", ""),
            "Status":       r.get("status", ""),
            "Score":        fo.get("overall_assessment_score", "N/A"),
            "Damages":      fo.get("confirmed_damage_count", "N/A"),
            "Cost (USD)":   (
                f"${fo.get('financial_estimate', {}).get('total_repair_usd', 0):,.0f}"
                if fo else "N/A"
            ),
            "Fraud Status": fr.get("status", "N/A"),
            "Trust":        fr.get("trust_score", "N/A"),
            "Ruling":       fo.get("claim_ruling_code", "N/A"),
            "Created":      (r.get("created_at", "") or "")[:16].replace("T", " "),
        })
    cols = ["Case ID", "Vehicle ID", "Status", "Score", "Damages",
            "Cost (USD)", "Fraud Status", "Trust", "Ruling", "Created"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Case Explorer handlers
# ─────────────────────────────────────────────────────────────────────────────

def _search_cases(vid_filter, status_filter, fraud_only):
    filters = {
        "vehicle_id": vid_filter or "",
        "status":     status_filter if status_filter != "All" else "",
        "is_fraud":   True if fraud_only else None,
    }
    return build_stats_html(), _cases_to_df(db_find(filters, limit=100))


def _load_case_detail(df: pd.DataFrame, evt: gr.SelectData):
    try:
        cid = str(df.iloc[evt.index[0]]["Case ID"]).rstrip("…").rstrip(".")
    except Exception:
        return "Select a row in the table above.", None, "", "", ""

    all_recs = db_find({}, limit=200)
    rec      = next(
        (r for r in all_recs if r.get("case_id", "").startswith(cid)), {}
    ) or db_get(cid)

    fo  = rec.get("final_output") or {}
    fr  = rec.get("fraud_report") or {}
    ins = rec.get("insurance")    or {}
    ud  = rec.get("user_data")    or {}
    img = (ud.get("image_paths") or [None])[0]

    damages  = fo.get("damages", [])
    dmg_txt  = ""
    for d in damages:
        vs   = d.get("verification_status", "?")
        conf = vs in _CONF
        icon = "✅" if conf else ("🚩" if d.get("rejected") else "❓")
        dmg_txt += (
            f"{icon} [{d.get('detection_id', '?')}] {d.get('type', '?')} @ "
            f"{d.get('location', '?')} — "
            f"Sev:{d.get('severity_gemini', d.get('severity', '?'))} "
            f"Conf:{d.get('confidence', 0):.2f}\n"
        )

    detail = (
        f"━━ CASE: {rec.get('case_id', 'N/A')} ━━\n"
        f"Vehicle ID    : {rec.get('vehicle_id', 'N/A')}\n"
        f"Owner         : {ud.get('owner_name', 'N/A')}\n"
        f"Status        : {rec.get('status', 'N/A')}\n"
        f"Created       : {(rec.get('created_at', '') or '')[:19]}\n\n"
        f"━━ ANALYSIS ━━\n"
        f"Health Score  : {fo.get('overall_assessment_score', 'N/A')}/100\n"
        f"Ruling        : [{fo.get('claim_ruling_code', 'N/A')}] "
        f"{fo.get('processing_status', 'N/A')}\n"
        f"Confirmed Dmg : {fo.get('confirmed_damage_count', 'N/A')}\n"
        f"Cost USD      : ${fo.get('financial_estimate', {}).get('total_repair_usd', 0):,.2f}\n"
        f"Cost INR      : {fo.get('financial_estimate', {}).get('total_repair_inr_fmt', 'N/A')}\n\n"
        f"━━ FRAUD ━━\n"
        f"Status        : {fr.get('status', 'N/A')}\n"
        f"Trust Score   : {fr.get('trust_score', 'N/A')}/100\n"
        f"Flags         : {len(fr.get('flags', []))}\n"
        + "\n".join(f"  • {f}" for f in fr.get("flags", []))
        + f"\n\n━━ INSURANCE ━━\n"
        f"Filed Claim   : {ins.get('filing_claim', 'N/A')}\n"
        f"Policy        : {ins.get('policy_number', 'N/A')}\n"
        f"Reason        : {ins.get('claim_reason', 'N/A')}\n\n"
        f"━━ DAMAGE DETECTIONS ({len(damages)}) ━━\n"
        + (dmg_txt or "None recorded.")
    )

    fo_json = (
        _json.dumps(fo, indent=2, default=str)[:3000] if fo
        else "No final output (fraud case)."
    )
    aud_rev = rec.get("auditor_review") or {}
    aud_txt = (
        f"Decision : {aud_rev.get('decision', '—')}\n"
        f"Note     : {aud_rev.get('note', '—')}\n"
        f"Reviewed : {(aud_rev.get('reviewed_at', '') or '')[:19]}"
        if aud_rev else "No auditor review yet."
    )
    return detail, img, fo_json, aud_txt, rec.get("case_id", "")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Insurance Claims handlers
# ─────────────────────────────────────────────────────────────────────────────

def _load_claims(status_filter):
    q      = "" if status_filter == "All" else (status_filter or "")
    recs   = db_find({"status": q}, limit=100)
    filed  = [r for r in recs if (r.get("insurance") or {}).get("filing_claim")]
    rows   = []
    for r in filed:
        ins = r.get("insurance") or {}
        fo  = r.get("final_output") or {}
        fr  = r.get("fraud_report") or {}
        rows.append({
            "Case ID":      (r.get("case_id", ""))[:20],
            "Vehicle":      r.get("vehicle_id", ""),
            "Policy No":    ins.get("policy_number", "N/A"),
            "Filed At":     (ins.get("submitted_at", "") or "")[:16].replace("T", " "),
            "Claim Reason": (ins.get("claim_reason", "") or "")[:40],
            "Cost (USD)":   (
                f"${fo.get('financial_estimate', {}).get('total_repair_usd', 0):,.0f}"
                if fo else "N/A"
            ),
            "Cost (INR)":   (
                (fo.get("financial_estimate") or {}).get("total_repair_inr_fmt", "N/A")
                if fo else "N/A"
            ),
            "Ruling":   fo.get("claim_ruling_code", "N/A"),
            "Status":   r.get("status", ""),
            "Fraud":    fr.get("status", "N/A"),
        })
    cols = ["Case ID", "Vehicle", "Policy No", "Filed At", "Claim Reason",
            "Cost (USD)", "Cost (INR)", "Ruling", "Status", "Fraud"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    summary = (
        f"**Total filed claims:** {len(rows)} | "
        f"**Approved:** {sum(1 for r in rows if r['Status'] == 'approved')} | "
        f"**Rejected:** {sum(1 for r in rows if r['Status'] == 'rejected')} | "
        f"**Pending:** {sum(1 for r in rows if r['Status'] not in ('approved', 'rejected'))}"
    )
    return summary, df


def _row_select_caseid(df: pd.DataFrame, evt: gr.SelectData) -> str:
    try:
        return str(df.iloc[evt.index[0]]["Case ID"]).rstrip("…").rstrip(".")
    except Exception:
        return ""


def _process_claim(case_id: str, decision: str) -> str:
    if not case_id or not case_id.strip():
        return "⚠️ Please enter a Case ID."
    rec = db_get(case_id.strip()) or {}
    if not rec:
        all_recs = db_find({}, limit=500)
        rec = next(
            (r for r in all_recs if r.get("case_id", "").startswith(case_id.strip())),
            {},
        )
    if not rec:
        return f"❌ Case '{case_id.strip()}' not found."
    if rec.get("is_fraud") and decision == "approved":
        return "❌ Cannot approve a fraud-flagged case. Clear it in Fraud Review first."
    db_upsert(
        rec["case_id"],
        status=decision,
        auditor_review={
            "decision":    decision.upper(),
            "note":        f"Auditor {decision} via Insurance Claims tab",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return f"✅ Case {rec['case_id'][:24]} → {decision.upper()} recorded."


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Fraud Review handlers
# ─────────────────────────────────────────────────────────────────────────────

def _load_fraud_cases():
    recs  = db_find({"is_fraud": True}, limit=100)
    more  = db_find({"status": "rejected"}, limit=50)
    seen  = {r.get("case_id") for r in recs}
    for r in more:
        if r.get("case_id") not in seen:
            recs.append(r); seen.add(r.get("case_id"))
    rows = []
    for r in recs:
        fr = r.get("fraud_report") or {}
        d  = fr.get("details", {})
        rows.append({
            "Case ID":      (r.get("case_id", ""))[:20],
            "Vehicle":      r.get("vehicle_id", ""),
            "Trust Score":  fr.get("trust_score", "N/A"),
            "Fraud Status": fr.get("status", "N/A"),
            "Flags":        len(fr.get("flags", [])),
            "pHash Match":  d.get("phash_check",         {}).get("status",       "N/A"),
            "ELA Score":    d.get("ai_generation_check",  {}).get("ela_score",    "N/A"),
            "Screen":       str(d.get("screen_detection", {}).get("is_screen",   "N/A")),
            "Auditor":      (r.get("auditor_review") or {}).get("decision", "—"),
            "Created":      (r.get("created_at", "") or "")[:16].replace("T", " "),
        })
    cols = ["Case ID", "Vehicle", "Trust Score", "Fraud Status", "Flags",
            "pHash Match", "ELA Score", "Screen", "Auditor", "Created"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _fraud_detail(df: pd.DataFrame, evt: gr.SelectData):
    try:
        cid = str(df.iloc[evt.index[0]]["Case ID"]).rstrip("…")
    except Exception:
        return "Click a row to load fraud detail.", ""
    all_recs = db_find({}, limit=500)
    rec = next((r for r in all_recs if r.get("case_id", "").startswith(cid)), {})
    if not rec:
        rec = db_get(cid)
    fr = rec.get("fraud_report") or {}
    d  = fr.get("details", {})
    ai = d.get("ai_generation_check", {})
    sc = d.get("screen_detection",    {})
    ph = d.get("phash_check",         {})
    txt = (
        f"━━ FRAUD DETAIL: {rec.get('case_id', 'N/A')} ━━\n\n"
        f"Trust Score   : {fr.get('trust_score', 'N/A')}/100\n"
        f"Status        : {fr.get('status', 'N/A')}\n"
        f"Checks Run    : {fr.get('checks_run', 'N/A')}\n"
        f"Checked At    : {(fr.get('checked_at', '') or '')[:19]}\n\n"
        f"━━ FLAGS ━━\n"
        + "\n".join(f"  • {f}" for f in fr.get("flags", []))
        + f"\n\n━━ pHASH CHECK ━━\n"
        f"Status        : {ph.get('status', 'N/A')}\n"
        f"pHash         : {ph.get('phash', 'N/A')}\n"
        f"Matched Claim : {ph.get('matched_claim', 'N/A')}\n"
        f"Hamming Dist  : {ph.get('hamming_distance', 'N/A')}\n\n"
        f"━━ AI-GENERATION (ELA) ━━\n"
        f"Is AI-Gen     : {ai.get('is_ai_generated', 'N/A')}\n"
        f"AI Probability: {ai.get('ai_probability', 'N/A')}\n"
        f"ELA Score     : {ai.get('ela_score', 'N/A')}\n"
        f"Method        : {ai.get('method', 'N/A')}\n\n"
        f"━━ SCREEN DETECTION ━━\n"
        f"Is Screen     : {sc.get('is_screen', 'N/A')}\n"
        f"Confidence    : {sc.get('confidence', 'N/A')}\n"
        f"Signals       : {', '.join(sc.get('signals', []))}\n\n"
        f"━━ AUDITOR REVIEW ━━\n"
        + (_json.dumps(rec.get("auditor_review") or {}, indent=2) or "None yet.")
    )
    return txt, rec.get("case_id", "")


def _mark_decision(case_id_state, decision, note):
    if not case_id_state or not case_id_state.strip():
        return "⚠️ No case selected — click a row first.", _load_fraud_cases()
    db_mark_auditor(case_id_state.strip(), decision, note or "")
    return (
        f"✅ Case {case_id_state[:20]} marked: {decision}",
        _load_fraud_cases(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — User Management handlers
# ─────────────────────────────────────────────────────────────────────────────

def _load_users():
    recs  = db_find({}, limit=500)
    users: dict = {}
    for r in recs:
        vid = r.get("vehicle_id", "unknown")
        if vid not in users:
            users[vid] = {
                "Vehicle / User": vid, "Cases": 0, "Claims Filed": 0,
                "Fraud Flags": 0, "Total Cost USD": 0.0,
                "Approved": 0, "Rejected": 0, "Last Activity": "",
            }
        u = users[vid]
        u["Cases"] += 1
        if (r.get("insurance") or {}).get("filing_claim"):
            u["Claims Filed"] += 1
        if r.get("is_fraud"):
            u["Fraud Flags"] += 1
        fin = (r.get("financial_estimate")
               or (r.get("final_output") or {}).get("financial_estimate", {})
               or {})
        u["Total Cost USD"] += fin.get("total_repair_usd", 0) or 0
        if r.get("status") == "approved": u["Approved"] += 1
        if r.get("status") == "rejected": u["Rejected"] += 1
        ts = r.get("updated_at") or r.get("created_at", "")
        if ts > u["Last Activity"]: u["Last Activity"] = ts[:16].replace("T", " ")
    rows = list(users.values())
    for row in rows:
        row["Total Cost USD"] = f"${row['Total Cost USD']:,.0f}"
    cols = ["Vehicle / User", "Cases", "Claims Filed", "Fraud Flags",
            "Total Cost USD", "Approved", "Rejected", "Last Activity"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    return f"**Unique vehicles / users:** {len(users)}", df


def _user_history(df: pd.DataFrame, evt: gr.SelectData):
    try:
        vid = df.iloc[evt.index[0]]["Vehicle / User"]
    except Exception:
        return "Click a user row to see claim history.", pd.DataFrame()
    recs = db_find({"vehicle_id": vid}, limit=50)
    rows = []
    for r in recs:
        fo = r.get("final_output") or {}
        rows.append({
            "Case ID":   (r.get("case_id", ""))[:20],
            "Status":    r.get("status", ""),
            "Score":     fo.get("overall_assessment_score", "N/A"),
            "Ruling":    fo.get("claim_ruling_code", "N/A"),
            "Cost (USD)": (
                f"${fo.get('financial_estimate', {}).get('total_repair_usd', 0):,.0f}"
                if fo else "N/A"
            ),
            "Fraud":   (r.get("fraud_report") or {}).get("status", "N/A"),
            "Created": (r.get("created_at", "") or "")[:16].replace("T", " "),
        })
    cols = ["Case ID", "Status", "Score", "Ruling", "Cost (USD)", "Fraud", "Created"]
    df_h = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    return f"Claim history for **{vid}** — {len(rows)} case(s):", df_h


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Audit Logs handler
# ─────────────────────────────────────────────────────────────────────────────

def _load_logs(vid_filter):
    recs = db_find({"vehicle_id": vid_filter or ""}, limit=50)
    if not recs:
        return "No records found.", "No records found.", pd.DataFrame()

    rows = []
    for r in recs:
        for agent, entry in (r.get("agent_trace") or {}).items():
            rows.append({
                "Case ID":  (r.get("case_id", ""))[:16],
                "Agent":    agent,
                "Decision": str(entry.get("decision", ""))[:60],
                "Timestamp": (entry.get("timestamp", "") or "")[:19],
                "Reasoning": (entry.get("reasoning", "") or "")[:80],
            })

    cols = ["Case ID", "Agent", "Decision", "Timestamp", "Reasoning"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    # Checkpoint dump for most recent case
    rec     = recs[0]
    chk     = rec.get("checkpoint_dump") or []
    chk_txt = (
        f"━━ CHECKPOINT DUMP: {rec.get('case_id', '?')} ━━\n"
        f"{'Step':>5}  {'Node':<10}  {'Timestamp':<22}  "
        f"{'Retries':>7}  {'Health':>8}  {'Dets':>5}  {'Msgs':>5}\n"
        + "─" * 80 + "\n"
    )
    for step in sorted(chk, key=lambda x: x.get("step", 0)):
        chk_txt += (
            f"{step.get('step', '?'):>5}  "
            f"{step.get('node', '?'):<10}  "
            f"{str(step.get('timestamp', ''))[:19]:<22}  "
            f"{str(step.get('retry_count', '?')):>7}  "
            f"{str(step.get('health_score', '?')):>8}  "
            f"{str(step.get('n_detections', '?')):>5}  "
            f"{str(step.get('n_messages', '?')):>5}\n"
        )

    full_trace = _json.dumps(
        rec.get("agent_trace") or {}, indent=2, default=str
    )[:4000]
    return chk_txt, full_trace, df


# ─────────────────────────────────────────────────────────────────────────────
# Auditor AI bot (Groq, scoped to all cases)
# ─────────────────────────────────────────────────────────────────────────────

def _auditor_bot(message: str, history: list):
    if not cfg.GROQ_ENABLED:
        return "", history + [[message, "⚠️ GROQ_API_KEY not set."]]
    try:
        recent  = db_find({}, limit=30)
        counts  = {}
        from src.db.mongo_client import db_count
        counts  = db_count()
        fraud_c = db_find({"is_fraud": True}, limit=10)
    except Exception:
        recent = fraud_c = []; counts = {}

    ctx_lines = [
        f"  [{(r.get('case_id','?'))[:14]}] veh={r.get('vehicle_id','?')} "
        f"status={r.get('status','?')} "
        f"score={((r.get('final_output') or {}).get('overall_assessment_score','N/A'))} "
        f"fraud={((r.get('fraud_report') or {}).get('status','N/A'))}"
        for r in recent[:15]
    ]
    system = (
        "You are SmartForge AI Auditor Assistant with full dashboard visibility.\n"
        "Answer concisely using only the provided data. Never fabricate case IDs.\n\n"
        f"STATS: {counts}\n\nRECENT CASES:\n" + "\n".join(ctx_lines or ["(none)"])
        + "\nFRAUD CASES:\n"
        + "\n".join(
            f"  [{(r.get('case_id','?'))[:14]}] "
            f"trust={((r.get('fraud_report') or {}).get('trust_score','N/A'))}"
            for r in fraud_c[:5]
        ) or "(none)"
    )
    try:
        from groq import Groq
        client = Groq(api_key=cfg.GROQ_API_KEY)
        msgs   = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                msgs.append({"role": "user",      "content": str(turn[0])})
                msgs.append({"role": "assistant",  "content": str(turn[1])})
        msgs.append({"role": "user", "content": message.strip()})
        reply = Groq(api_key=cfg.GROQ_API_KEY).chat.completions.create(
            model=cfg.GROQ_MODEL, messages=msgs,
            max_tokens=600, temperature=0.25,
        ).choices[0].message.content.strip()
        return "", history + [[message, reply]]
    except Exception as exc:
        return "", history + [[message, f"⚠️ Bot error: {str(exc)[:160]}"]]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_auditor_demo() -> gr.Blocks:
    """Construct and return the 5-tab Auditor Gradio Blocks app."""

    with gr.Blocks(title="SmartForge — Auditor Dashboard") as demo:

        with gr.Sidebar(open=False, position="right"):
            gr.Markdown("### 🤖 SmartForge AI Auditor")
            gr.Markdown("Ask about cases, fraud flags, pipeline results, or system stats.")
            bot_chat   = gr.Chatbot(label="System", show_label=False, height=500)
            bot_msg    = gr.Textbox(show_label=False, placeholder="Type your query…")
            bot_submit = gr.Button("Send", variant="primary")
            bot_submit.click(fn=_auditor_bot, inputs=[bot_msg, bot_chat],
                             outputs=[bot_msg, bot_chat])
            bot_msg.submit(fn=_auditor_bot,   inputs=[bot_msg, bot_chat],
                           outputs=[bot_msg, bot_chat])

        gr.HTML(get_css_block())
        _sel_case = gr.State(value="")

        gr.HTML(
            "<div style='background:linear-gradient(135deg,#b71c1c,#c62828);"
            "color:white;padding:20px 28px;border-radius:12px;margin-bottom:8px;'>"
            "<div style='font-size:22px;font-weight:800;'>"
            "🔒 SmartForge — Auditor / Admin Dashboard</div>"
            f"<div style='font-size:12px;opacity:.8;margin-top:4px;'>"
            f"Role: AUDITOR — Full case visibility · No user-ID filter applied"
            f" &nbsp;|&nbsp; DB: {db_backend_info()} &nbsp;|&nbsp; "
            f"{cfg.GRADIO_VERSION_TAG}</div></div>"
        )

        with gr.Tabs(selected=0) as aud_tabs:

            # ── Tab 1: Case Explorer ──────────────────────────────────────────
            with gr.TabItem("🗂️ 1 · Case Explorer", id=0):
                gr.HTML("<p class='tab-desc'>Search all cases. Click any row for full detail.</p>")
                with gr.Row():
                    a1_vid    = gr.Textbox(label="Vehicle ID (partial match)",
                                           placeholder="e.g. VH001", max_lines=1, scale=2)
                    a1_status = gr.Dropdown(
                        label="Status Filter",
                        choices=["All", "uploaded", "analyzed", "claim_submitted",
                                 "fraud_flagged", "approved", "rejected"],
                        value="All", scale=1,
                    )
                    a1_fraud  = gr.Checkbox(label="🚨 Fraud Only", value=False, scale=1)
                    a1_search = gr.Button("🔍 Search", variant="primary", scale=1)

                a1_stats   = gr.HTML(value=build_stats_html())
                a1_results = gr.Dataframe(
                    headers=["Case ID", "Vehicle ID", "Status", "Score", "Damages",
                              "Cost (USD)", "Fraud Status", "Trust", "Ruling", "Created"],
                    datatype=["str"] * 10, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(10, "fixed"),
                )
                gr.Markdown("##### 📋 Case Detail")
                with gr.Row():
                    with gr.Column(scale=2):
                        a1_detail    = gr.Textbox(label="Summary", lines=18,
                                                   interactive=False, show_copy_button=True)
                    with gr.Column(scale=1):
                        a1_img       = gr.Image(label="Vehicle Photo", height=220,
                                                interactive=False)
                        a1_aud_rev   = gr.Textbox(label="Auditor Review", lines=4,
                                                   interactive=False)
                a1_fo_json = gr.Code(label="final_output JSON (truncated)",
                                      language="json", interactive=False)

                a1_search.click(fn=_search_cases,
                                inputs=[a1_vid, a1_status, a1_fraud],
                                outputs=[a1_stats, a1_results])
                a1_results.select(fn=_load_case_detail, inputs=[a1_results],
                                  outputs=[a1_detail, a1_img, a1_fo_json,
                                           a1_aud_rev, _sel_case])

            # ── Tab 2: Insurance Claims ───────────────────────────────────────
            with gr.TabItem("📋 2 · Insurance Claims", id=1):
                gr.HTML("<p class='tab-desc'>All submitted insurance claims.</p>")
                with gr.Row():
                    a2_status = gr.Dropdown(
                        label="Filter by Status",
                        choices=["All", "claim_submitted", "approved", "rejected"],
                        value="All",
                    )
                    a2_load = gr.Button("🔄 Load Claims", variant="primary")
                a2_summary = gr.Markdown()
                a2_table   = gr.Dataframe(
                    headers=["Case ID", "Vehicle", "Policy No", "Filed At", "Claim Reason",
                              "Cost (USD)", "Cost (INR)", "Ruling", "Status", "Fraud"],
                    datatype=["str"] * 10, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(10, "fixed"),
                )
                gr.Markdown("##### ⚖️ Process a Claim")
                with gr.Row():
                    a2_target  = gr.Textbox(label="Case ID to Process",
                                            placeholder="e.g. VH001-abc123",
                                            max_lines=1, scale=3)
                    a2_approve = gr.Button("✅ Approve Claim", variant="primary", scale=1)
                    a2_reject  = gr.Button("❌ Reject Claim",  variant="stop",    scale=1)
                a2_action = gr.Textbox(label="Action Result", lines=3, interactive=False)

                a2_status.change(fn=_load_claims, inputs=[a2_status],
                                 outputs=[a2_summary, a2_table])
                a2_load.click(fn=_load_claims,   inputs=[a2_status],
                              outputs=[a2_summary, a2_table])
                a2_table.select(fn=_row_select_caseid, inputs=[a2_table],
                                outputs=[a2_target])
                a2_approve.click(
                    fn=lambda sid: _process_claim(sid, "approved"),
                    inputs=[a2_target], outputs=[a2_action],
                )
                a2_reject.click(
                    fn=lambda sid: _process_claim(sid, "rejected"),
                    inputs=[a2_target], outputs=[a2_action],
                )

            # ── Tab 3: Fraud Review ───────────────────────────────────────────
            with gr.TabItem("🚨 3 · Fraud Review", id=2):
                gr.HTML("<p class='tab-desc'>All fraud-flagged cases. "
                        "Click a row to load forensic detail.</p>")
                a3_load  = gr.Button("🔄 Load Fraud Cases", variant="primary")
                a3_table = gr.Dataframe(
                    headers=["Case ID", "Vehicle", "Trust Score", "Fraud Status", "Flags",
                              "pHash Match", "ELA Score", "Screen", "Auditor", "Created"],
                    datatype=["str"] * 10, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(10, "fixed"),
                )
                gr.Markdown("##### 🔬 Forensic Detail")
                a3_detail = gr.Textbox(label="Full Fraud Analysis", lines=18,
                                        interactive=False, show_copy_button=True)
                gr.Markdown("##### ⚖️ Auditor Decision")
                with gr.Row():
                    a3_decision = gr.Radio(
                        label="Mark as:",
                        choices=["Confirm Fraud", "Clear — Not Fraud",
                                 "Approve Claim", "Reject Claim"],
                        value="Confirm Fraud",
                    )
                    a3_note = gr.Textbox(label="Note (optional)",
                                          placeholder="Add reasoning for audit trail…",
                                          max_lines=2, scale=2)
                a3_mark_btn = gr.Button("💾 Save Auditor Decision", variant="primary")
                a3_mark_msg = gr.Textbox(label="Result", lines=2, interactive=False)

                a3_load.click(fn=_load_fraud_cases, outputs=[a3_table])
                a3_table.select(fn=_fraud_detail, inputs=[a3_table],
                                outputs=[a3_detail, _sel_case])
                a3_mark_btn.click(fn=_mark_decision,
                                  inputs=[_sel_case, a3_decision, a3_note],
                                  outputs=[a3_mark_msg, a3_table])

            # ── Tab 4: User Management ────────────────────────────────────────
            with gr.TabItem("👤 4 · User Management", id=3):
                gr.HTML("<p class='tab-desc'>Per-vehicle summary. "
                        "Click a row to see full claim history.</p>")
                a4_load    = gr.Button("🔄 Load Users", variant="primary")
                a4_summary = gr.Markdown()
                a4_users   = gr.Dataframe(
                    headers=["Vehicle / User", "Cases", "Claims Filed", "Fraud Flags",
                              "Total Cost USD", "Approved", "Rejected", "Last Activity"],
                    datatype=["str"] * 8, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(8, "fixed"),
                )
                gr.Markdown("##### 📜 Claim History for Selected User")
                a4_hist_hdr = gr.Markdown()
                a4_hist     = gr.Dataframe(
                    headers=["Case ID", "Status", "Score", "Ruling",
                              "Cost (USD)", "Fraud", "Created"],
                    datatype=["str"] * 7, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(7, "fixed"),
                )
                a4_load.click(fn=_load_users, outputs=[a4_summary, a4_users])
                a4_users.select(fn=_user_history, inputs=[a4_users],
                                outputs=[a4_hist_hdr, a4_hist])

            # ── Tab 5: Audit Logs ─────────────────────────────────────────────
            with gr.TabItem("📊 5 · Audit Logs", id=4):
                gr.HTML("<p class='tab-desc'>Agent reasoning trace, MemorySaver checkpoint "
                        "dump, and decision timeline — the compliance backbone.</p>")
                with gr.Row():
                    a5_vid  = gr.Textbox(label="Vehicle ID Filter (blank = latest 50)",
                                          placeholder="e.g. VH001", max_lines=1)
                    a5_load = gr.Button("🔄 Load Logs", variant="primary")
                gr.Markdown("##### 📌 MemorySaver Checkpoint Timeline")
                a5_chk   = gr.Textbox(label="Checkpoint Dump (most recent case)",
                                       lines=12, interactive=False,
                                       show_copy_button=True)
                gr.Markdown("##### 🧠 Agent Trace (full pipeline reasoning)")
                a5_trace = gr.Code(label="agent_trace JSON", language="json",
                                    interactive=False)
                gr.Markdown("##### 🗂️ All Agent Decisions")
                a5_table = gr.Dataframe(
                    headers=["Case ID", "Agent", "Decision", "Timestamp", "Reasoning"],
                    datatype=["str"] * 5, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(5, "fixed"),
                )
                a5_load.click(fn=_load_logs, inputs=[a5_vid],
                              outputs=[a5_chk, a5_trace, a5_table])

        # Auto-load on page open
        demo.load(fn=lambda: _search_cases("", "All", False),
                  outputs=[a1_stats, a1_results])
        demo.load(fn=lambda: _load_claims("All"),
                  outputs=[a2_summary, a2_table])
        demo.load(fn=_load_fraud_cases, outputs=[a3_table])

        def _tab_refresh(evt: gr.SelectData):
            if evt.index == 0:
                return _search_cases("", "All", False)
            return gr.update(), gr.update()

        aud_tabs.select(fn=_tab_refresh, outputs=[a1_stats, a1_results])

        gr.HTML(
            f"<div style='font-size:11px;color:var(--sf-text-muted);text-align:center;"
            f"margin-top:10px;padding-top:8px;border-top:1px solid var(--sf-border);'>"
            f"SmartForge Auditor Dashboard {cfg.GRADIO_VERSION_TAG}"
            f" · DB: {db_backend_info()} · Role: AUDITOR (full access)<br>"
            f"<span style='color:#c0392b;font-size:10px;'>"
            f"RESTRICTED — authorised auditors only.</span></div>"
        )

    try:
        demo.theme = get_theme(cfg.GRADIO_THEME)
    except Exception:
        pass

    return demo
