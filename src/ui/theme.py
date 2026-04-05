"""
SmartForge — UI Theme & Badge Helpers
=======================================
Single source of truth for every visual element shared between the User
Dashboard and the Auditor Dashboard.

Exports
-------
    SMARTFORGE_CSS          str  — full CSS injected via gr.HTML('<style>…</style>')
    get_theme(name)         → gr.themes.*   — Gradio theme factory
    score_badge(score)      → str  — coloured health-score HTML badge
    ruling_badge(code, …)   → str  — claim ruling HTML badge
    fraud_badge(fr)         → str  — fraud status HTML badge
    stat_card(label, value) → str  — dashboard stat card HTML

Design approach
---------------
All colours use CSS custom properties (``--sf-*``) defined in ``:root`` for
light mode and overridden in ``.dark`` for dark mode.  This means every badge
and card automatically adapts when the user switches the Gradio theme without
any Python-side logic.

The CSS is injected inside the Gradio Blocks context as:
    gr.HTML(f"<style>{SMARTFORGE_CSS}</style>")
"""

import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# Gradio compatibility shim
# ─────────────────────────────────────────────────────────────────────────────

import gradio.components as _gr_comp
if not hasattr(_gr_comp, "LoginButton"):
    class _LoginButtonStub:
        pass
    _gr_comp.LoginButton = _LoginButtonStub


# ─────────────────────────────────────────────────────────────────────────────
# Master CSS (CSS custom properties + component styles)
# ─────────────────────────────────────────────────────────────────────────────

SMARTFORGE_CSS = """
/* ── CSS Custom Properties — Light defaults ──────────────────────────────── */
:root {
  --sf-bg-page:#f8fafc; --sf-bg-card:#ffffff; --sf-bg-card-alt:#f1f5f9;
  --sf-bg-inset:#f0f4ff; --sf-border:#e2e8f0; --sf-border-focus:#3b82f6;
  --sf-text-primary:#0f172a; --sf-text-secondary:#374151;
  --sf-text-muted:#6b7280; --sf-text-hint:#94a3b8;
  --sf-brand:#1a237e; --sf-brand-light:#e8eaf6; --sf-brand-border:#7986cb;
  --sf-ok-bg:#d1fae5;  --sf-ok-brd:#6ee7b7;  --sf-ok-txt:#065f46;
  --sf-warn-bg:#fef3c7;--sf-warn-brd:#fcd34d;--sf-warn-txt:#92400e;
  --sf-err-bg:#fee2e2; --sf-err-brd:#fca5a5; --sf-err-txt:#7f1d1d;
  --sf-info-bg:#dbeafe;--sf-info-brd:#93c5fd;--sf-info-txt:#1e40af;
  --sf-neu-bg:#f1f5f9; --sf-neu-brd:#cbd5e1; --sf-neu-txt:#334155;
  --sf-scroll-track:#f1f5f9;--sf-scroll-thumb:#cbd5e1;--sf-scroll-hover:#94a3b8;
  --sf-shadow-sm:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  --sf-shadow-md:0 4px 12px rgba(0,0,0,.10),0 2px 4px rgba(0,0,0,.06);
  --sf-trans:200ms cubic-bezier(.4,0,.2,1);
  --sf-node-done-bg:#dcfce7;--sf-node-done-brd:#4ade80;--sf-node-done-txt:#166534;
  --sf-node-idle-bg:#f8fafc;--sf-node-idle-brd:#e2e8f0;--sf-node-idle-txt:#94a3b8;
}
/* ── Dark mode overrides ─────────────────────────────────────────────────── */
.dark,:is(.dark *):root,body.dark,html.dark,[data-theme="dark"],
.gradio-container.dark {
  --sf-bg-page:#0f172a; --sf-bg-card:#1e293b; --sf-bg-card-alt:#1a2744;
  --sf-bg-inset:#1e3a5f; --sf-border:#334155; --sf-border-focus:#60a5fa;
  --sf-text-primary:#f1f5f9; --sf-text-secondary:#cbd5e1;
  --sf-text-muted:#94a3b8; --sf-text-hint:#475569;
  --sf-brand:#818cf8; --sf-brand-light:#1e1b4b; --sf-brand-border:#6366f1;
  --sf-ok-bg:#064e3b;  --sf-ok-brd:#059669;  --sf-ok-txt:#a7f3d0;
  --sf-warn-bg:#451a03;--sf-warn-brd:#d97706;--sf-warn-txt:#fde68a;
  --sf-err-bg:#450a0a; --sf-err-brd:#dc2626; --sf-err-txt:#fca5a5;
  --sf-info-bg:#1e3a5f;--sf-info-brd:#3b82f6;--sf-info-txt:#93c5fd;
  --sf-neu-bg:#1e293b; --sf-neu-brd:#475569; --sf-neu-txt:#cbd5e1;
  --sf-scroll-track:#1e293b;--sf-scroll-thumb:#475569;--sf-scroll-hover:#64748b;
  --sf-shadow-sm:0 1px 3px rgba(0,0,0,.3);
  --sf-shadow-md:0 4px 12px rgba(0,0,0,.4);
  --sf-node-done-bg:#052e16;--sf-node-done-brd:#22c55e;--sf-node-done-txt:#86efac;
  --sf-node-idle-bg:#1e293b;--sf-node-idle-brd:#334155;--sf-node-idle-txt:#475569;
}
/* ── Global ──────────────────────────────────────────────────────────────── */
footer{display:none !important;}
.tabitem{position:relative !important;z-index:1 !important;}
/* ── Custom scrollbars ───────────────────────────────────────────────────── */
::-webkit-scrollbar{width:7px;height:7px}
::-webkit-scrollbar-track{background:var(--sf-scroll-track);border-radius:99px}
::-webkit-scrollbar-thumb{background:var(--sf-scroll-thumb);border-radius:99px;
  border:2px solid var(--sf-scroll-track);transition:background var(--sf-trans)}
::-webkit-scrollbar-thumb:hover{background:var(--sf-scroll-hover)}
*{scrollbar-width:thin;scrollbar-color:var(--sf-scroll-thumb) var(--sf-scroll-track)}
/* ── Textarea / Textbox ──────────────────────────────────────────────────── */
.gr-textbox textarea,textarea.scroll-hide,.block.padded textarea{
  font-size:13px !important;line-height:1.65 !important;
  color:var(--sf-text-primary) !important;
  background:var(--sf-bg-card) !important;
  border-color:var(--sf-border) !important;
  border-radius:8px !important;padding:10px 14px !important;
  overflow-y:auto !important;resize:vertical !important;
  min-height:80px !important;}
textarea::placeholder{color:var(--sf-text-hint) !important;}
textarea[rows="18"],textarea[rows="20"]{max-height:460px !important;}
/* ── Labels ──────────────────────────────────────────────────────────────── */
label.block span,.gr-block label,span.svelte-1b6s6vi{
  color:var(--sf-text-secondary) !important;
  font-weight:600 !important;font-size:13px !important;}
/* ── Dataframe ───────────────────────────────────────────────────────────── */
.gr-dataframe{max-height:480px !important;overflow-y:auto !important;
  overflow-x:auto !important;border:1px solid var(--sf-border) !important;
  border-radius:8px !important;display:block !important;}
.gr-dataframe table{border-collapse:collapse;width:100%;}
.gr-dataframe thead{position:sticky;top:0;z-index:3;}
.gr-dataframe th{background:var(--sf-bg-card-alt) !important;
  color:var(--sf-text-primary) !important;font-weight:700 !important;
  padding:10px 12px !important;border-bottom:2px solid var(--sf-border) !important;
  white-space:nowrap;cursor:pointer;user-select:none;
  transition:background var(--sf-trans);}
.gr-dataframe th:hover{background:var(--sf-bg-inset) !important;}
.sf-sort-asc::after{content:" ▲";font-size:10px;opacity:.8;}
.sf-sort-desc::after{content:" ▼";font-size:10px;opacity:.8;}
.sf-sort-none::after{content:" ⇅";font-size:10px;opacity:.35;}
.gr-dataframe td{color:var(--sf-text-secondary) !important;
  padding:9px 12px !important;border-bottom:1px solid var(--sf-border) !important;
  vertical-align:top;}
.gr-dataframe tr:hover td{background:var(--sf-bg-inset) !important;}
/* ── Code blocks ─────────────────────────────────────────────────────────── */
.gr-code{overflow:auto !important;max-height:440px !important;
  border-radius:8px !important;border:1px solid var(--sf-border) !important;}
.gr-code code{font-size:12px !important;line-height:1.6 !important;}
/* ── Markdown ────────────────────────────────────────────────────────────── */
.prose,.prose p,.gr-markdown,.gr-markdown p{color:var(--sf-text-secondary) !important;}
.prose h5,.gr-markdown h5{color:var(--sf-text-primary) !important;
  font-size:14px !important;font-weight:700 !important;
  margin:12px 0 6px 0 !important;}
/* ── Chat ────────────────────────────────────────────────────────────────── */
.chatbot{max-height:55vh !important;overflow-y:auto !important;
  border-radius:10px !important;}
.chatbot .message{border-radius:12px !important;font-size:13px !important;
  padding:10px 14px !important;line-height:1.6 !important;}
/* ── Badges ──────────────────────────────────────────────────────────────── */
.sf-badge-wrap{border-radius:10px;padding:12px 16px;border:1px solid;
  box-shadow:var(--sf-shadow-sm);transition:box-shadow var(--sf-trans);}
.sf-badge-wrap:hover{box-shadow:var(--sf-shadow-md);}
.sf-badge-title{font-weight:700;font-size:15px;margin-bottom:4px;}
.sf-badge-sub{font-size:13px;margin-top:3px;}
.sf-badge-reason{font-size:12px;margin-top:3px;opacity:.85;}
.sf-flag-list{margin:6px 0 0 16px;padding:0;}
.sf-flag-item{font-size:12px;margin-bottom:3px;}
/* ── Health score badge ──────────────────────────────────────────────────── */
.sf-score-wrap{display:inline-flex;align-items:center;border-radius:24px;
  padding:10px 24px;border:2px solid;box-shadow:var(--sf-shadow-md);}
.sf-score-val{font-size:38px;font-weight:800;line-height:1;}
.sf-score-den{font-size:14px;margin-left:6px;opacity:.8;}
/* ── Pipeline timeline nodes ─────────────────────────────────────────────── */
.sf-node{display:inline-flex;flex-direction:column;align-items:center;
  border:1px solid;border-radius:10px;padding:6px 10px;margin:3px;min-width:72px;
  transition:all var(--sf-trans);box-shadow:var(--sf-shadow-sm);position:relative;}
.sf-node:hover{box-shadow:var(--sf-shadow-md);transform:translateY(-1px);}
.sf-node-icon{font-size:18px;}
.sf-node-label{font-size:9px;font-weight:600;text-align:center;
  margin-top:4px;line-height:1.35;}
/* ── Status stepper ──────────────────────────────────────────────────────── */
.sf-step{flex:1;text-align:center;padding:8px 4px;border-bottom:3px solid;
  font-size:11px;transition:all var(--sf-trans);}
/* ── Stat cards (auditor dashboard) ──────────────────────────────────────── */
.sf-stat-card{flex:1;border-radius:12px;padding:18px 20px;
  border:1px solid var(--sf-border);box-shadow:var(--sf-shadow-sm);
  background:var(--sf-bg-card);text-align:center;
  transition:all var(--sf-trans);}
.sf-stat-card:hover{box-shadow:var(--sf-shadow-md);transform:translateY(-2px);}
.sf-stat-value{font-size:30px;font-weight:800;line-height:1.1;}
.sf-stat-label{font-size:12px;font-weight:500;margin-top:5px;
  color:var(--sf-text-muted);}
/* ── Info / warn / success boxes ─────────────────────────────────────────── */
.sf-info-box{border-radius:10px;padding:14px 18px;margin-bottom:10px;
  font-size:13px;border:1px solid;
  background:var(--sf-info-bg);border-color:var(--sf-info-brd);
  color:var(--sf-info-txt);}
.sf-warn-box{border-radius:10px;padding:12px 16px;margin-bottom:10px;
  font-size:13px;border:1px solid;
  background:var(--sf-warn-bg);border-color:var(--sf-warn-brd);
  color:var(--sf-warn-txt);}
.sf-success-box{border-radius:10px;padding:14px 18px;margin-top:14px;
  font-size:13px;border:1px solid;
  background:var(--sf-ok-bg);border-color:var(--sf-ok-brd);color:var(--sf-ok-txt);}
.sf-tip-box{font-size:11px;background:var(--sf-bg-card-alt);
  border:1px solid var(--sf-border);border-radius:8px;
  padding:10px 12px;margin-top:6px;line-height:1.9;
  color:var(--sf-text-secondary);}
/* ── Tab description ──────────────────────────────────────────────────────── */
.tab-desc{font-size:13px;color:var(--sf-text-muted);margin-bottom:10px;}
/* ── Claim form body ──────────────────────────────────────────────────────── */
.claim-form-body{border:1px solid var(--sf-border);border-radius:10px;
  padding:16px 18px;margin-top:8px;background:var(--sf-bg-card);}
/* ── Map wrap ────────────────────────────────────────────────────────────── */
#sf_map_wrap{position:relative;z-index:10;overflow:hidden;border-radius:8px;
  border:1px solid var(--sf-border) !important;}
.gr-form,.gr-panel{overflow:visible !important;}
/* ── Dark mode sync script guard ─────────────────────────────────────────── */
.gradio-container>footer{display:none !important;}
"""

# Dark-mode sync JS (injected alongside CSS)
DARKMODE_SYNC_JS = """
<script>
(function(){
  function sync(){
    var d=document.body.classList.contains('dark')
       ||document.documentElement.classList.contains('dark')
       ||(document.querySelector('.gradio-container')||{}).classList?.contains('dark');
    if(d){document.documentElement.classList.add('dark');
          document.body.classList.add('dark');}
    else{document.documentElement.classList.remove('dark');
         document.body.classList.remove('dark');}
  }
  sync();
  var o=new MutationObserver(sync);
  o.observe(document.body,{attributes:true,subtree:true,attributeFilter:['class']});
  o.observe(document.documentElement,{attributes:true,attributeFilter:['class']});
})();
</script>
"""

# Click-to-sort JS for gr.Dataframe tables
SORTABLE_TABLE_JS = """
<script>
(function(){
  var SF_SORT='sf-sort';
  function sortTable(tbody,colIdx,dir){
    var rows=Array.from(tbody.querySelectorAll('tr'));
    rows.sort(function(a,b){
      var av=(a.cells[colIdx]?a.cells[colIdx].innerText.trim():'');
      var bv=(b.cells[colIdx]?b.cells[colIdx].innerText.trim():'');
      var an=parseFloat(av.replace(/[^0-9.\-]/g,'')),
          bn=parseFloat(bv.replace(/[^0-9.\-]/g,''));
      var cmp=(!isNaN(an)&&!isNaN(bn))?(an-bn)
              :av.localeCompare(bv,undefined,{numeric:true,sensitivity:'base'});
      return dir==='asc'?cmp:-cmp;
    });
    rows.forEach(function(r){tbody.appendChild(r);});
  }
  function attachSort(table){
    if(table.dataset[SF_SORT])return;
    table.dataset[SF_SORT]='1';
    var thead=table.querySelector('thead');
    if(!thead)return;
    var ths=Array.from(thead.querySelectorAll('th'));
    ths.forEach(function(th,i){
      th.classList.add('sf-sort-none');
      th.addEventListener('click',function(){
        var tbody=table.querySelector('tbody');
        if(!tbody)return;
        var cur=th.dataset.sfDir||'none';
        var next=(cur==='none'||cur==='desc')?'asc':'desc';
        ths.forEach(function(t){
          t.classList.remove('sf-sort-asc','sf-sort-desc','sf-sort-none');
          t.classList.add('sf-sort-none');delete t.dataset.sfDir;});
        th.classList.remove('sf-sort-none');
        th.classList.add(next==='asc'?'sf-sort-asc':'sf-sort-desc');
        th.dataset.sfDir=next;sortTable(tbody,i,next);
      });
    });
  }
  function wireAll(){document.querySelectorAll('.gr-dataframe table').forEach(attachSort);}
  var obs=new MutationObserver(function(muts){
    muts.forEach(function(m){
      m.addedNodes.forEach(function(n){
        if(!n.querySelectorAll)return;
        n.querySelectorAll('.gr-dataframe table').forEach(attachSort);
      });
    });
  });
  function init(){wireAll();obs.observe(document.body,{childList:true,subtree:true});}
  if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}
  else{init();}
})();
</script>
"""


def get_css_block() -> str:
    """Return the combined CSS + JS HTML block for injection via gr.HTML()."""
    return f"<style>{SMARTFORGE_CSS}</style>{SORTABLE_TABLE_JS}{DARKMODE_SYNC_JS}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio theme factory
# ─────────────────────────────────────────────────────────────────────────────

def get_theme(name: str = "soft"):
    """
    Return a Gradio theme object by name.
    Supported: soft | default | monochrome | ocean | citrus | origin | base
    Falls back to soft on any error.
    """
    _map = {
        "soft":       gr.themes.Soft,
        "default":    gr.themes.Default,
        "monochrome": gr.themes.Monochrome,
        "ocean":      gr.themes.Ocean,
        "citrus":     gr.themes.Citrus,
        "origin":     gr.themes.Origin,
        "base":       gr.themes.Base,
    }
    try:
        return _map.get(name.lower(), gr.themes.Soft)()
    except Exception:
        return gr.themes.Soft()


# ─────────────────────────────────────────────────────────────────────────────
# Badge / card HTML builders
# ─────────────────────────────────────────────────────────────────────────────

def score_badge(score) -> str:
    """Return a coloured circular health-score HTML badge."""
    try:
        val = int(float(str(score)))
    except Exception:
        return "<span style='color:var(--sf-text-muted);font-size:14px;'>N/A</span>"

    if val >= 80:
        col, bg, brd = "var(--sf-ok-txt)",   "var(--sf-ok-bg)",   "var(--sf-ok-brd)"
    elif val >= 60:
        col, bg, brd = "var(--sf-warn-txt)", "var(--sf-warn-bg)", "var(--sf-warn-brd)"
    else:
        col, bg, brd = "var(--sf-err-txt)",  "var(--sf-err-bg)",  "var(--sf-err-brd)"

    return (
        f"<div class='sf-score-wrap' style='background:{bg};border-color:{brd};'>"
        f"<span class='sf-score-val' style='color:{col};'>{val}</span>"
        f"<span class='sf-score-den' style='color:{col};'>/100</span>"
        f"</div>"
    )


def ruling_badge(code: str, status: str, ruling: str = "") -> str:
    """Return a coloured claim-ruling HTML badge."""
    _pal = {
        "CLM_APPROVED": ("var(--sf-ok-txt)",   "var(--sf-ok-bg)",   "var(--sf-ok-brd)",   "✅"),
        "CLM_WORKSHOP": ("var(--sf-warn-txt)",  "var(--sf-warn-bg)", "var(--sf-warn-brd)", "🔧"),
        "CLM_MANUAL":   ("var(--sf-err-txt)",   "var(--sf-err-bg)",  "var(--sf-err-brd)",  "👁️"),
        "CLM_PENDING":  ("var(--sf-info-txt)",  "var(--sf-info-bg)", "var(--sf-info-brd)", "⏳"),
    }
    col, bg, brd, icon = _pal.get(
        code,
        ("var(--sf-neu-txt)", "var(--sf-neu-bg)", "var(--sf-neu-brd)", "ℹ️"),
    )
    ruling_html = (
        f"<div class='sf-badge-reason' style='color:{col};'>{ruling}</div>"
        if ruling else ""
    )
    return (
        f"<div class='sf-badge-wrap' style='background:{bg};border-color:{brd};'>"
        f"<div class='sf-badge-title' style='color:{col};'>{icon} [{code}]</div>"
        f"<div class='sf-badge-sub'   style='color:{col};'>{status}</div>"
        f"{ruling_html}</div>"
    )


def fraud_badge(fr: dict | None) -> str:
    """Return a coloured fraud-status HTML badge."""
    if not fr:
        return (
            "<span style='color:var(--sf-text-muted);font-size:13px;'>"
            "Fraud check not run.</span>"
        )
    status = fr.get("status", "N/A")
    score  = fr.get("trust_score", "N/A")
    flags  = fr.get("flags", [])

    if "SUSPICIOUS" in str(status):
        col, bg, brd, icon = (
            "var(--sf-err-txt)", "var(--sf-err-bg)", "var(--sf-err-brd)", "🚨"
        )
    elif "VERIFIED" in str(status) or "BYPASSED" in str(status):
        col, bg, brd, icon = (
            "var(--sf-ok-txt)", "var(--sf-ok-bg)", "var(--sf-ok-brd)", "✅"
        )
    else:
        col, bg, brd, icon = (
            "var(--sf-neu-txt)", "var(--sf-neu-bg)", "var(--sf-neu-brd)", "ℹ️"
        )

    flags_html = "".join(
        f"<li class='sf-flag-item' style='color:{col};'>• {f}</li>"
        for f in flags
    )
    return (
        f"<div class='sf-badge-wrap' style='background:{bg};border-color:{brd};'>"
        f"<div class='sf-badge-title' style='color:{col};'>"
        f"{icon} {status} — Trust Score: {score}/100</div>"
        + (
            f"<ul class='sf-flag-list'>{flags_html}</ul>"
            if flags_html else ""
        )
        + "</div>"
    )


def stat_card(label: str, value, color: str = "var(--sf-brand)") -> str:
    """Return a single stat card HTML block (used in the Auditor Dashboard)."""
    return (
        f"<div class='sf-stat-card'>"
        f"<div class='sf-stat-value' style='color:{color};'>{value}</div>"
        f"<div class='sf-stat-label'>{label}</div>"
        f"</div>"
    )
