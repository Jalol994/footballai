"""Look & feel: CSS, flags, HTML snippets (single-line HTML so Markdown never turns it into code)."""
import hashlib, html, re

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(1100px 500px at 8% -8%,#12384f 0%,transparent 60%),#0b0f19;}
section[data-testid="stSidebar"]{background:#0e1424;border-right:1px solid #1c2740;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1150px;}
.hero{background:linear-gradient(135deg,#0f766e,#1d4ed8 60%,#7c3aed);border-radius:18px;padding:20px 24px;margin-bottom:16px;box-shadow:0 10px 30px rgba(0,0,0,.45)}
.hero h1{margin:0;font-size:1.6rem;color:#fff;font-weight:800}
.hero p{margin:4px 0 0;color:#dbeafe;font-size:.92rem}
.card{background:#131a2b;border:1px solid #1f2b47;border-radius:14px;padding:14px 16px;margin-bottom:8px}
.card .lbl{color:#8aa0c8;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em}
.card .val{color:#fff;font-size:1.7rem;font-weight:800;line-height:1.2}
.match{display:flex;align-items:center;justify-content:space-between;gap:8px;background:#131a2b;border:1px solid #1f2b47;border-radius:16px;padding:16px 10px;margin:8px 0 6px}
.team{flex:1;text-align:center}
.flag{font-size:2.6rem;line-height:1.15}
.init{display:inline-flex;width:52px;height:52px;border-radius:50%;align-items:center;justify-content:center;font-weight:800;color:#fff;font-size:1.05rem}
.nm{font-weight:700;color:#fff;margin-top:4px}
.xg{color:#8aa0c8;font-size:.85rem}
.vs{font-weight:800;color:#5b6b8c}
.rbar{display:flex;height:12px;border-radius:8px;overflow:hidden;margin:4px 0 2px}
.rlab{display:flex;justify-content:space-between;color:#c9d4ea;font-size:.82rem;margin-bottom:12px}
.btitle{font-weight:700;color:#fff;margin:6px 0 6px}
.brow{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:.86rem;color:#c9d4ea}
.brow .k{width:24px;text-align:right;color:#8aa0c8}
.btrack{flex:1;height:10px;background:#1a2440;border-radius:6px;overflow:hidden}
.bfill{height:100%;background:#3b82f6;border-radius:6px}
.bfill.top{background:linear-gradient(90deg,#14b8a6,#22d3ee)}
.brow .v{width:46px}
.warn{background:#3a2e0b;border:1px solid #6b5514;color:#fde68a;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:.88rem}
.disc{color:#7e8db0;font-size:.8rem;margin-top:14px}
.slip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0}.sbox{background:linear-gradient(160deg,#134e4a,#1e3a8a);border-radius:14px;padding:12px 6px;text-align:center}.sl{font-size:.72rem;color:#bfdbfe}.sv{font-size:2.4rem;font-weight:800;color:#fff;line-height:1.1}.sp{color:#99f6e4;font-size:.85rem}
@media (max-width:640px){.slip{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.2rem}.flag{font-size:2rem}.card .val{font-size:1.3rem}.init{width:42px;height:42px}}
"""

ICONS = {"dashboard": "🏠", "upcoming": "📅", "analysis": "🔎", "betman": "🎯", "exact": "🔢",
         "odds": "💱", "teams": "👥", "backtest": "🧪", "perf": "📊", "sources": "⚙️"}


def _cc(code):
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in code.upper())


def _tag(s):  # England / Scotland / Wales sub-national flags
    return "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in s) + "\U000E007F"


ENG, SCO, WAL = _tag("gbeng"), _tag("gbsct"), _tag("gbwls")
LEAGUE_FLAGS = {
    "E0": ENG, "E1": ENG, "E2": ENG, "E3": ENG, "EC": ENG, "SC0": SCO, "SC1": SCO,
    "D1": _cc("DE"), "D2": _cc("DE"), "SP1": _cc("ES"), "SP2": _cc("ES"), "I1": _cc("IT"), "I2": _cc("IT"),
    "F1": _cc("FR"), "F2": _cc("FR"), "N1": _cc("NL"), "B1": _cc("BE"), "P1": _cc("PT"), "T1": _cc("TR"), "G1": _cc("GR"),
    # API-Football league ids
    "39": ENG, "40": ENG, "140": _cc("ES"), "78": _cc("DE"), "135": _cc("IT"), "61": _cc("FR"), "88": _cc("NL"),
    "94": _cc("PT"), "203": _cc("TR"), "235": _cc("RU"), "292": _cc("KR"), "98": _cc("JP"), "253": _cc("US"), "71": _cc("BR"),
}
_C = dict(uzbekistan="UZ", russia="RU", kazakhstan="KZ", tajikistan="TJ", kyrgyzstan="KG", turkmenistan="TM", azerbaijan="AZ",
          georgia="GE", armenia="AM", ukraine="UA", poland="PL", spain="ES", germany="DE", france="FR", italy="IT", brazil="BR",
          argentina="AR", portugal="PT", netherlands="NL", belgium="BE", croatia="HR", turkey="TR", turkiye="TR", usa="US",
          mexico="MX", iran="IR", australia="AU", china="CN", qatar="QA", uruguay="UY", colombia="CO", morocco="MA",
          senegal="SN", switzerland="CH", denmark="DK", sweden="SE", norway="NO", austria="AT", greece="GR", serbia="RS",
          egypt="EG", nigeria="NG", canada="CA", chile="CL", ecuador="EC", ireland="IE", japan="JP", india="IN", iraq="IQ")
COUNTRY_FLAGS = {k: _cc(v) for k, v in _C.items()}
COUNTRY_FLAGS.update({"england": ENG, "scotland": SCO, "wales": WAL, "south korea": _cc("KR"), "korea republic": _cc("KR"),
                      "saudi arabia": _cc("SA"), "united arab emirates": _cc("AE"), "czech republic": _cc("CZ"),
                      "united states": _cc("US"), "russian federation": _cc("RU"), "uae": _cc("AE")})


def team_flag(team, league=None):
    """National team name -> its flag; otherwise the flag of the club's league country; else ''."""
    f = COUNTRY_FLAGS.get(str(team).strip().lower())
    if f: return f
    return LEAGUE_FLAGS.get(str(league), "") if league is not None else ""


def _initials(name):
    p = re.findall(r"\w+", name)
    return "".join(x[0] for x in p[:2]).upper() or "?"


def crest(name, flag):
    if flag:
        return f'<div class="flag">{flag}</div>'
    hue = int(hashlib.md5(name.encode()).hexdigest()[:6], 16) % 360
    return f'<div class="init" style="background:hsl({hue},55%,40%)">{html.escape(_initials(name))}</div>'


def hero(title, sub):
    return f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>'


def card(label, value):
    return f'<div class="card"><div class="lbl">{html.escape(str(label))}</div><div class="val">{html.escape(str(value))}</div></div>'


def warn(text):
    return f'<div class="warn">{html.escape(text)}</div>'


def disclaimer(text):
    return f'<div class="disc">{html.escape(text)}</div>'


def match_banner(h, a, hf, af, lh, la, p, labels):
    ph, pdw, pa = (p["home"] * 100, p["draw"] * 100, p["away"] * 100)
    team = lambda n, f, x: f'<div class="team">{crest(n, f)}<div class="nm">{html.escape(n)}</div><div class="xg">xG {x:.2f}</div></div>'
    bar = (f'<div class="rbar"><div style="width:{ph:.1f}%;background:#14b8a6"></div><div style="width:{pdw:.1f}%;background:#64748b"></div>'
           f'<div style="width:{pa:.1f}%;background:#6366f1"></div></div>'
           f'<div class="rlab"><span>{labels[0]} {ph:.0f}%</span><span>{labels[1]} {pdw:.0f}%</span><span>{labels[2]} {pa:.0f}%</span></div>')
    return f'<div class="match">{team(h, hf, lh)}<div class="vs">VS</div>{team(a, af, la)}</div>{bar}'


def bars(title, cats, probs):
    top = max(range(len(probs)), key=lambda i: probs[i])
    rows = "".join(f'<div class="brow"><span class="k">{c}</span><div class="btrack"><div class="bfill{" top" if i == top else ""}" '
                   f'style="width:{max(p * 100, 0.4):.1f}%"></div></div><span class="v">{p * 100:.1f}%</span></div>'
                   for i, (c, p) in enumerate(zip(cats, probs)))
    return f'<div class="card"><div class="btitle">{html.escape(title)}</div>{rows}</div>'


def slip(cells):
    """cells: [(label, value, prob_percent)] -> big prediction-slip boxes."""
    items = "".join(f'<div class="sbox"><div class="sl">{html.escape(l)}</div><div class="sv">{html.escape(str(v))}</div><div class="sp">{p:.0f}%</div></div>'
                    for l, v, p in cells)
    return f'<div class="slip">{items}</div>'
