"""
SmartForge — User Dashboard (5-Tab)
=====================================
Port 7860 · Audience: Vehicle owner / claimant

Tab flow
--------
Tab 1  📥 Vehicle Intake
    Vehicle ID (mandatory), owner name, vehicle type, multi-image upload,
    native HTML5 date picker, Leaflet map with autocomplete search + GPS.

Tab 2  🛡️ Insurance Preference
    Choose claim vs assessment-only BEFORE analysis runs.
    Saves filing_claim flag + policy/reason so fraud layer knows whether
    to activate.

Tab 3  🔬 Damage Analysis
    Triggers the full LangGraph pipeline.
    Shows detection table, pipeline timeline, primary photo.
    Navigates automatically to Tab 4 on completion.

Tab 4  📊 Executive Summary
    Health score badge, Total Loss / Repairable banner, line-item cost table,
    Groq executive summary, claim ruling badge, fraud badge, forensic text.

Tab 5  💬 AI Assistant
    Groq-powered chatbot scoped strictly to the current session.
    Chat history persisted to DB after each turn.
"""

import shutil
import traceback
import uuid
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from src.config.settings import cfg
from src.db.mongo_client import db_backend_info, db_get, db_upsert
from src.ui.helpers import (
    build_checkpoint_list,
    chat_with_session,
    extract_phash,
    pipeline_timeline,
    run_pipeline,
    status_stepper,
)
from src.ui.theme import (
    fraud_badge,
    get_css_block,
    get_theme,
    ruling_badge,
    score_badge,
)

# ── Empty DataFrames for initial render ───────────────────────────────────────
_E_DMG = pd.DataFrame(columns=["ID", "Type", "Location", "Severity", "Conf", "Status"])
_E_FIN = pd.DataFrame(columns=["Part", "Action", "Severity", "Cost (USD)", "Cost (INR)"])
_CONF  = {"confirmed", "gemini_golden_frame_confirmed", "gemini_confirmed"}


# ─────────────────────────────────────────────────────────────────────────────
# Tab handler functions
# ─────────────────────────────────────────────────────────────────────────────

def _handle_intake(
    vehicle_id, owner_name, vehicle_type, image_files,
    date_str, incident_lat, incident_lon,
):
    """Save vehicle data to DB and navigate to Tab 2."""
    errors = []
    if not vehicle_id or not vehicle_id.strip():
        errors.append("❌ Vehicle ID is mandatory.")
    if not image_files:
        errors.append("❌ At least one damage photo is required.")
    if errors:
        return "\n".join(errors), gr.update(selected=0), ""

    vid = vehicle_id.strip().upper()
    sid = f"{vid}-{uuid.uuid4().hex[:6]}"
    incident_date = (
        (date_str or "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    if isinstance(image_files, str):
        image_files = [image_files]
    elif not isinstance(image_files, list):
        image_files = [image_files]

    saved = []
    for i, fp in enumerate(image_files):
        if fp is None:
            continue
        fp_str = fp if isinstance(fp, str) else (
            fp.get("name", "") if isinstance(fp, dict) else str(fp)
        )
        dst = f"/content/{sid}_img{i}.jpg"
        try:
            shutil.copy(fp_str, dst)
            saved.append(dst)
        except Exception:
            saved.append(fp_str)

    user_data = {
        "vehicle_id":   vid,
        "owner_name":   owner_name or "Unknown",
        "vehicle_type": vehicle_type or "Auto-Detect",
        "image_paths":  saved,
        "incident_date": incident_date,
        "incident_lat":  float(incident_lat) if incident_lat else 0.0,
        "incident_lon":  float(incident_lon) if incident_lon else 0.0,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    db_upsert(
        sid, vehicle_id=vid, user_id=vid,
        status="uploaded", user_data=user_data,
        images=saved, is_fraud=False, fraud_attempts=0,
    )

    msg = (
        f"✅ Intake saved!\n\n"
        f"Session    : {sid}\n"
        f"Vehicle ID : {vid}\n"
        f"Images     : {len(saved)} file(s)\n"
        f"Date       : {incident_date}\n"
        f"Claimant   : {owner_name or '—'}\n"
        f"DB backend : {db_backend_info()}\n\n"
        "→ Switch to Insurance Preference (Tab 2) to proceed."
    )
    return msg, gr.update(selected=1), sid


def _handle_insurance_save(
    session_id, want, policy_num, claim_reason, date_override, notes,
):
    """Save insurance preference + claim details then navigate to Tab 3."""
    if not session_id:
        return "⚠️ Complete Tab 1 (Vehicle Intake) first.", "", gr.update(selected=0)

    filing = (want == "Yes – I want to file a claim")
    rec    = db_get(session_id)
    ud     = rec.get("user_data") or {}
    ud["wants_insurance"] = filing

    ins = {
        "filing_claim":       filing,
        "preference_saved_at": datetime.now(timezone.utc).isoformat(),
    }

    if filing:
        if not policy_num or not policy_num.strip():
            return "❌ Policy Number is mandatory.", "", gr.update()
        if not claim_reason or not claim_reason.strip():
            return "❌ Claim reason is mandatory.", "", gr.update()
        ins.update({
            "policy_number": policy_num.strip(),
            "claim_reason":  claim_reason.strip(),
            "incident_date": date_override or ud.get("incident_date", ""),
            "notes":         notes or "",
        })

    db_upsert(session_id, user_data=ud, status="pref_saved", insurance=ins)

    msg = (
        "✅ Insurance claim details saved.\n\n"
        "🛡️ Fraud detection will run automatically during Damage Analysis.\n"
        "→ Proceed to Tab 3 (Damage Analysis)."
        if filing else
        "ℹ️ Assessment-only mode saved.\n\n"
        "Fraud checks bypassed.\n→ Proceed to Tab 3 (Damage Analysis)."
    )
    return msg, "", gr.update(selected=2)


def _handle_analysis(session_id):
    """Run the full LangGraph pipeline for the session's images."""
    if not session_id:
        return (
            "⚠️ No session — complete Tab 1 first.",
            None, _E_DMG, "", "", gr.update(selected=0),
        )

    rec       = db_get(session_id)
    ud        = rec.get("user_data") or {}
    img_paths = ud.get("image_paths", [])
    if not img_paths:
        return (
            "⚠️ No images in session.",
            None, _E_DMG, "", "", gr.update(selected=2),
        )

    # 3-strike fraud guard
    fraud_attempts = int(rec.get("fraud_attempts") or 0)
    if fraud_attempts >= cfg.MAX_FRAUD_RETRIES:
        return (
            f"🚫 CASE PERMANENTLY CLOSED\n\n"
            f"Maximum fraud tolerance reached ({cfg.MAX_FRAUD_RETRIES}/{cfg.MAX_FRAUD_RETRIES}).\n"
            "Please start a new case.",
            None, _E_DMG, "", "", gr.update(selected=2),
        )

    wants_insurance = ud.get("wants_insurance", False)
    bypass = cfg.BYPASS_FRAUD or not wants_insurance

    try:
        fo, partial = run_pipeline(
            image_path   = img_paths[0],
            vehicle_id   = ud.get("vehicle_id", "VH-UI"),
            policy_id    = "POL-PENDING",
            claim_date   = ud.get("incident_date", ""),
            claim_lat    = ud.get("incident_lat", 0.0),
            claim_lon    = ud.get("incident_lon", 0.0),
            bypass_fraud = bypass,
        )
    except Exception:
        tb = traceback.format_exc()
        return (
            f"❌ Pipeline error:\n{tb[:600]}",
            None, _E_DMG, "", "", gr.update(selected=2),
        )

    fraud_rep   = partial.get("fraud_report") or {}
    is_fraud    = fraud_rep.get("status", "").startswith("SUSPICIOUS")
    chk_dump    = build_checkpoint_list(partial)
    agent_trace = fo.get("pipeline_trace", partial.get("pipeline_trace", {}))

    if is_fraud:
        fraud_attempts += 1
        retries_left    = cfg.MAX_FRAUD_RETRIES - fraud_attempts
        db_upsert(
            session_id,
            status="fraud_flagged",
            fraud_attempts=fraud_attempts,
            fraud_report=fraud_rep,
            fraud_hash=extract_phash(fraud_rep),
            agent_trace=agent_trace,
            is_fraud=True,
        )
        msg = (
            f"🚨 IMAGE FLAGGED AS FRAUDULENT\n\n"
            f"Trust Score  : {fraud_rep.get('trust_score', 0)}/100\n"
            f"Flags        : {len(fraud_rep.get('flags', []))}\n"
            f"Attempt      : {fraud_attempts}/{cfg.MAX_FRAUD_RETRIES}\n\n"
            + (
                f"❌ Tolerance limit reached. Case PERMANENTLY CLOSED."
                if retries_left <= 0 else
                f"⚠️  You have {retries_left} retry attempt(s) remaining.\n"
                "Return to Tab 1 with a legitimate image."
            )
        )
        stepper = status_stepper("fraud_flagged" if retries_left > 0 else "rejected")
        return msg, None, _E_DMG, "", stepper, gr.update(selected=2)

    # Successful analysis
    db_upsert(
        session_id,
        status="analyzed",
        final_output=fo if fo else None,
        checkpoint_dump=chk_dump,
        fraud_report=fraud_rep,
        fraud_hash=extract_phash(fraud_rep),
        agent_trace=agent_trace,
        is_fraud=False,
    )

    damages = fo.get("damages") or partial.get("damages_output") or []
    rows = []
    for d in damages:
        vs   = d.get("verification_status", "?")
        conf = vs in _CONF and not d.get("rejected", False)
        rows.append({
            "ID":       d.get("detection_id", "?"),
            "Type":     d.get("type", "?"),
            "Location": d.get("location", "?"),
            "Severity": d.get("severity_gemini") or d.get("severity", "?"),
            "Conf":     f"{d.get('confidence', 0):.2f}",
            "Status": (
                "✅ Confirmed" if conf else
                ("🚩 Rejected" if d.get("rejected") else f"❓ {vs}")
            ),
        })
    df_dmg = pd.DataFrame(rows) if rows else _E_DMG

    js      = fo.get("job_summary", {})
    tl_html = pipeline_timeline(js.get("agents_run", []), js.get("retry_count", 0))
    stepper = status_stepper("analyzed")
    n_conf  = fo.get(
        "confirmed_damage_count",
        sum(1 for r in rows if "✅" in r.get("Status", "")),
    )
    msg = (
        f"✅ Analysis complete!\n\n"
        f"Confirmed  : {n_conf} damage(s)\n"
        f"Score      : {fo.get('overall_assessment_score', 'N/A')}/100\n"
        f"Disposition: {fo.get('financial_estimate', {}).get('disposition', 'N/A')}\n"
        f"Elapsed    : {js.get('elapsed_seconds', 'N/A')}s\n\n"
        "→ Proceed to the Executive Summary tab (Tab 4)."
    )
    return msg, img_paths[0], df_dmg, tl_html, stepper, gr.update(selected=3)


def _handle_summary_load(session_id):
    """Load and render the Executive Summary for a completed session."""
    _empty = (
        "⚠️ Run analysis first.",
        "",
        score_badge("—"),
        _E_FIN,
        "",
        "",
        "",
        "",
    )
    if not session_id:
        return _empty

    rec       = db_get(session_id)
    fo        = rec.get("final_output") or {}
    fraud_rep = rec.get("fraud_report") or {}
    ud        = rec.get("user_data")    or {}
    status    = rec.get("status", "uploaded")
    stepper   = status_stepper(status)

    if not fo:
        return (
            f"⚠️ No final output — may be a fraud-flagged case.\n\n"
            f"Fraud status : {fraud_rep.get('status', 'N/A')}\n"
            f"Trust score  : {fraud_rep.get('trust_score', 'N/A')}/100",
            stepper, score_badge("—"), _E_FIN,
            "", fraud_badge(fraud_rep), "", "",
        )

    owner  = ud.get("owner_name", "")
    exec_s = fo.get("executive_summary") or fo.get("ai_narrative_summary", "")
    if owner and owner != "Unknown":
        exec_s = f"Dear {owner},\n\n{exec_s}"

    fin        = fo.get("financial_estimate") or {}
    total_loss = fin.get("total_loss_flag", False)
    total_usd  = fin.get("total_repair_usd", 0)
    total_inr  = fin.get("total_repair_inr_fmt", "N/A")
    disp       = fin.get("disposition", "N/A")

    tl_banner = (
        "<div style='background:linear-gradient(135deg,#c0392b,#e74c3c);"
        "color:white;padding:18px 24px;border-radius:10px;font-size:17px;"
        "font-weight:700;text-align:center;margin-bottom:10px;'>"
        f"⚠️ TOTAL LOSS RECOMMENDED ⚠️<br>"
        f"<span style='font-size:13px;font-weight:400;'>${total_usd:,.0f} USD / {total_inr} "
        f"exceeds {int(cfg.TOTAL_LOSS_THRESHOLD * 100)}% of vehicle value "
        f"(${cfg.VEHICLE_VALUE:,} USD)</span></div>"
        if total_loss else
        "<div style='background:linear-gradient(135deg,#27ae60,#2ecc71);"
        "color:white;padding:14px 24px;border-radius:10px;font-size:15px;"
        "font-weight:600;text-align:center;margin-bottom:10px;'>"
        f"✅ Repairable — {disp} | ${total_usd:,.0f} USD / {total_inr}</div>"
    )

    items = fin.get("line_items", [])
    rows  = [
        {
            "Part":       it.get("part",    "?"),
            "Action":     it.get("action",  "?"),
            "Severity":   it.get("severity","?"),
            "Cost (USD)": f"${it.get('cost_usd', 0):,.2f}",
            "Cost (INR)": it.get("cost_inr_fmt", "?"),
        }
        for it in items
    ]
    if rows:
        rows.append({
            "Part": "── TOTAL ──", "Action": disp, "Severity": "—",
            "Cost (USD)": f"${total_usd:,.2f}", "Cost (INR)": total_inr,
        })
    df_fin = pd.DataFrame(rows) if rows else _E_FIN

    ruling_html = ruling_badge(
        fo.get("claim_ruling_code", ""),
        fo.get("processing_status",  "N/A"),
        fo.get("claim_ruling",       ""),
    )
    forensic = fo.get("forensic_report", "Not available.")

    return (
        exec_s, stepper,
        score_badge(fo.get("overall_assessment_score", "N/A")),
        df_fin, ruling_html,
        fraud_badge(fraud_rep), forensic, tl_banner,
    )


def _toggle_claim_section(want):
    visible = (want == "Yes – I want to file a claim")
    return gr.update(visible=visible), "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_demo() -> gr.Blocks:
    """Construct and return the 5-tab User Gradio Blocks app."""

    with gr.Blocks(title=f"{cfg.GRADIO_APP_TITLE} · User") as demo:

        gr.HTML(get_css_block())
        sid_state = gr.State(value="")

        gr.HTML(
            f"<div style='background:linear-gradient(135deg,#1a237e,#283593);"
            f"color:white;padding:20px 28px;border-radius:12px;margin-bottom:8px;'>"
            f"<div style='font-size:22px;font-weight:800;'>{cfg.GRADIO_APP_TITLE}</div>"
            f"<div style='font-size:12px;opacity:.8;margin-top:4px;'>"
            f"{cfg.GRADIO_APP_SUBTITLE} · {cfg.GRADIO_VERSION_TAG}"
            f" &nbsp;|&nbsp; DB: {db_backend_info()}</div></div>"
        )

        with gr.Tabs(selected=0) as u_tabs:

            # ── Tab 1: Vehicle Intake ─────────────────────────────────────────
            with gr.TabItem("📥 1 · Vehicle Intake", id=0):
                gr.HTML("<p class='tab-desc'>Enter vehicle info and upload damage photos. "
                        "<b>Vehicle ID and at least one image are mandatory.</b></p>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🚗 Vehicle Details")
                        t1_vid   = gr.Textbox(label="Vehicle ID *(mandatory)*",
                                              placeholder="e.g. VH001 or TN-09-AB-1234", max_lines=1)
                        t1_owner = gr.Textbox(label="Owner / Claimant Name",
                                              placeholder="e.g. Rajesh Kumar", max_lines=1)
                        t1_vtype = gr.Dropdown(
                            label="Vehicle Type",
                            choices=["Auto-Detect (Gemini VLM)", "Car / Sedan / SUV",
                                     "2-Wheeler (Bike/Scooter)", "3-Wheeler (Auto)"],
                            value="Auto-Detect (Gemini VLM)")

                        gr.Markdown("#### 📅 Incident Date")
                        gr.HTML("""
<div id='sf_date_wrap' style='padding:4px 0 8px 0;'>
  <input type='date' id='sf_date_picker' style='width:100%;padding:10px 15px;
    font-size:15px;border:1px solid #dee2e6;border-radius:8px;
    background:rgba(255,255,255,0.05);color:inherit;color-scheme:dark light;
    cursor:pointer;box-sizing:border-box;'>
  <script>
    (function(){var t=new Date().toISOString().split('T')[0];
    var i=document.getElementById('sf_date_picker');i.value=t;i.max=t;})();
  </script>
</div>""")
                        t1_date = gr.Textbox(
                            visible=False,
                            value=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        )

                        gr.Markdown("#### 📍 Incident Location")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9, min_width=0):
                                gr.HTML(
                                    "<div style='border:1px solid #dee2e6;border-radius:8px;"
                                    "overflow:hidden;height:320px;'>"
                                    "<iframe src='https://www.openstreetmap.org/export/embed.html"
                                    "?bbox=68,8,97,37&layer=mapnik' "
                                    "style='width:100%;height:100%;border:none;'></iframe></div>"
                                    "<p style='font-size:11px;color:#888;margin-top:4px;'>"
                                    "Enter coordinates manually below or use GPS.</p>"
                                )
                            with gr.Column(scale=2, min_width=200):
                                t1_lat = gr.Textbox(label="Latitude",
                                                    placeholder="e.g. 13.0827",
                                                    max_lines=1)
                                t1_lon = gr.Textbox(label="Longitude",
                                                    placeholder="e.g. 80.2707",
                                                    max_lines=1)

                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Upload Damage Photo(s)")
                        gr.HTML("<div class='sf-tip-box'>📌 <b>Tips:</b><br>"
                                "• Well-lit, clear photos improve detection<br>"
                                "• Multiple angles activate Batch 2 Multi-Image Map-Reduce<br>"
                                "• EXIF GPS metadata enables GPS consistency fraud check</div>")
                        t1_imgs = gr.File(
                            label="Drag & drop (JPG/PNG, multi-image supported)",
                            file_count="multiple", file_types=["image"],
                        )

                t1_btn    = gr.Button("→ Save & Proceed to Insurance Preference",
                                      variant="primary", size="lg")
                t1_status = gr.Textbox(label="✅ Intake Status", lines=6,
                                       interactive=False, show_copy_button=True,
                                       placeholder="Intake result will appear here…")
                t1_btn.click(
                    fn=_handle_intake,
                    inputs=[t1_vid, t1_owner, t1_vtype, t1_imgs,
                            t1_date, t1_lat, t1_lon],
                    outputs=[t1_status, u_tabs, sid_state],
                    js="""(vid,owner,vtype,imgs,date_hidden,lat,lon)=>{
  var dp=document.getElementById('sf_date_picker');
  var d=(dp&&dp.value)?dp.value:(date_hidden||'');
  return [vid,owner,vtype,imgs,d,lat,lon];}""",
                )

            # ── Tab 2: Insurance Preference ───────────────────────────────────
            with gr.TabItem("🛡️ 2 · Insurance Preference", id=1):
                gr.HTML("<p class='tab-desc'>Choose whether to file an insurance claim "
                        "<b>before</b> Damage Analysis runs.</p>")
                gr.HTML("<div class='sf-info-box'><b>🔒 Why this comes before analysis?</b><br>"
                        "Choosing <b>Yes</b> activates the full 5-check fraud detection layer "
                        "automatically. You have up to <b>3 attempts</b> if fraud flags are raised."
                        "</div>")
                t2_want = gr.Radio(
                    label="Do you want to file an insurance claim for this damage?",
                    choices=["Yes – I want to file a claim", "No – damage assessment only"],
                    value="No – damage assessment only",
                )
                with gr.Group(visible=False, elem_classes=["claim-form-body"]) as t2_claim_sec:
                    gr.HTML("<div class='sf-warn-box'>🛡️ <b>Fraud check will run automatically "
                            "during Damage Analysis.</b> You have up to "
                            f"<b>{cfg.MAX_FRAUD_RETRIES} attempts</b>.</div>")
                    with gr.Row():
                        t2_policy = gr.Textbox(label="Policy Number *(mandatory)*",
                                               placeholder="e.g. POL-2024-001",
                                               max_lines=1, scale=2)
                        t2_claim_date = gr.Textbox(label="Accident Date",
                                                   placeholder="Auto-filled or enter manually",
                                                   max_lines=1, scale=1)
                    t2_reason = gr.Textbox(label="Claim Reason *(mandatory)*",
                                           placeholder="e.g. Rear-end collision at NH-44",
                                           lines=2)
                    t2_notes  = gr.Textbox(label="Additional Notes (optional)",
                                           placeholder="FIR number, witness info…",
                                           lines=2)

                t2_btn    = gr.Button("✅ Save Preference & Proceed to Analysis",
                                      variant="primary", size="lg")
                t2_status = gr.Textbox(label="Submission Status", lines=4,
                                       interactive=False)
                t2_fraud  = gr.HTML()

                t2_want.change(
                    fn=_toggle_claim_section,
                    inputs=[t2_want],
                    outputs=[t2_claim_sec, t2_status, t2_fraud],
                )
                t2_btn.click(
                    fn=_handle_insurance_save,
                    inputs=[sid_state, t2_want, t2_policy, t2_reason,
                            t2_claim_date, t2_notes],
                    outputs=[t2_status, t2_fraud, u_tabs],
                )
                u_tabs.change(
                    fn=lambda sid: db_get(sid).get("user_data", {}).get("incident_date", "")
                    if sid else "",
                    inputs=[sid_state], outputs=[t2_claim_date],
                )

            # ── Tab 3: Damage Analysis ────────────────────────────────────────
            with gr.TabItem("🔬 3 · Damage Analysis", id=2):
                gr.HTML("<p class='tab-desc'>Runs the full AI pipeline: "
                        "SAHI → SAM → MiDaS → Gemini VLM → Golden Frame → LangGraph.</p>")
                t3_btn     = gr.Button("🔍 Run Full Analysis", variant="primary", size="lg")
                t3_stepper = gr.HTML()
                t3_status  = gr.Textbox(label="Pipeline Status", lines=8,
                                        interactive=False, show_copy_button=True,
                                        placeholder="Click Run Full Analysis to start…")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_img = gr.Image(label="Primary Vehicle Photo", height=260,
                                          interactive=False, show_download_button=True)
                    with gr.Column(scale=1):
                        gr.Markdown("##### ⚡ Pipeline Timeline")
                        t3_timeline = gr.HTML(
                            value="<div style='color:var(--sf-text-muted);padding:8px;"
                            "font-size:13px;'>▶ Run analysis to see agent timeline.</div>"
                        )
                gr.Markdown("##### 🔎 Detection Records")
                t3_dmg = gr.Dataframe(
                    headers=["ID", "Type", "Location", "Severity", "Conf", "Status"],
                    datatype=["str"] * 6, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(6, "fixed"),
                )
                gr.HTML("<div class='sf-success-box'>"
                        "✅ <b>Analysis complete?</b> &nbsp;→&nbsp; "
                        "Switch to <b>📊 4 · Executive Summary</b> for your full report."
                        "</div>")

                _t3_evt = t3_btn.click(
                    fn=_handle_analysis,
                    inputs=[sid_state],
                    outputs=[t3_status, t3_img, t3_dmg, t3_timeline, t3_stepper, u_tabs],
                )

            # ── Tab 4: Executive Summary ──────────────────────────────────────
            with gr.TabItem("📊 4 · Executive Summary", id=3):
                gr.HTML("<p class='tab-desc'>Full claim report — auto-loads after analysis, "
                        "or click Refresh.</p>")
                t4_refresh = gr.Button("🔄 Refresh Report", variant="secondary", size="sm")
                t4_stepper = gr.HTML()
                t4_tl      = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("##### 📝 Executive Summary")
                        t4_exec = gr.Textbox(lines=10, interactive=False,
                                             show_copy_button=True,
                                             placeholder="Executive summary appears after analysis…")
                    with gr.Column(scale=1):
                        gr.Markdown("##### 🏥 Health Score")
                        t4_health = gr.HTML(
                            value="<div style='padding:12px;color:#6b7280;font-size:13px;'>"
                            "Health score appears after analysis.</div>"
                        )
                        gr.Markdown("##### ⚖️ Claim Ruling")
                        t4_ruling = gr.HTML()
                gr.Markdown("##### 💰 Line-Item Repair Estimate")
                t4_fin = gr.Dataframe(
                    headers=["Part", "Action", "Severity", "Cost (USD)", "Cost (INR)"],
                    datatype=["str"] * 5, wrap=True, interactive=False,
                    row_count=(0, "dynamic"), col_count=(5, "fixed"),
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### 🛡️ Fraud Detection")
                        t4_fraud = gr.HTML()
                    with gr.Column(scale=1):
                        gr.Markdown("##### 🔬 Forensic Integrity")
                        t4_forensic = gr.Textbox(lines=5, interactive=False,
                                                  show_copy_button=True,
                                                  placeholder="Forensic report appears after analysis…")

                _summary_outs = [
                    t4_exec, t4_stepper, t4_health,
                    t4_fin, t4_ruling, t4_fraud, t4_forensic, t4_tl,
                ]
                t4_refresh.click(
                    fn=_handle_summary_load,
                    inputs=[sid_state], outputs=_summary_outs,
                )
                # Auto-load summary when analysis completes
                _t3_evt.then(
                    fn=_handle_summary_load,
                    inputs=[sid_state], outputs=_summary_outs,
                )

            # ── Tab 5: AI Assistant ───────────────────────────────────────────
            with gr.TabItem("💬 5 · AI Assistant", id=4):
                gr.HTML(
                    f"<p class='tab-desc'>Ask about <em>your</em> vehicle, damages, costs, "
                    f"or claim. Complete <b>Tab 1 → Tab 3</b> first.</p>"
                    f"<div style='font-size:11px;background:#e8eaf6;border-radius:6px;"
                    f"padding:8px 12px;margin-bottom:8px;color:#1a237e;'>"
                    f"🤖 <b>Groq</b> {cfg.GROQ_MODEL if cfg.GROQ_ENABLED else 'N/A'}"
                    f" &nbsp;·&nbsp; <b>Scope:</b> current session only"
                    f" &nbsp;·&nbsp; <b>Storage:</b> {db_backend_info()}</div>"
                )
                gr.ChatInterface(
                    fn=chat_with_session,
                    additional_inputs=[sid_state],
                    title="", description="",
                    examples=[
                        ["What damages were found on my vehicle?"],
                        ["What is my total repair cost in INR?"],
                        ["Should I file an insurance claim?"],
                        ["Explain the fraud detection result."],
                        ["What is my vehicle health score?"],
                    ],
                    submit_btn="Send", stop_btn="Stop",
                )

        gr.HTML(
            f"<div style='font-size:11px;color:var(--sf-text-muted);"
            f"text-align:center;margin-top:10px;padding-top:8px;"
            f"border-top:1px solid var(--sf-border);'>"
            f"SmartForge {cfg.GRADIO_VERSION_TAG} · LangGraph DCG · SAHI+SAM+MiDaS · "
            f"Gemini 2.5 Flash · Groq · Golden Frame · 5-Check Fraud Layer<br>"
            f"<span style='color:#c0392b;font-size:10px;'>"
            f"All outputs are AI-generated estimates — verify before settlement.</span></div>"
        )

    try:
        demo.theme = get_theme(cfg.GRADIO_THEME)
    except Exception:
        pass

    return demo
