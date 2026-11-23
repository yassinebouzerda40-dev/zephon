"""Serveur Zephon : site + IA (clés cachées ici) + connexion Google/GitHub + discussions sauvegardées.
Lancer :  python app.py   (puis ouvrir http://localhost:8000)"""
import os, json, sqlite3, secrets, time, asyncio
from datetime import date
from collections import defaultdict, deque
from pathlib import Path
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, JSONResponse
from starlette.exceptions import HTTPException as SHTTP
from starlette.middleware.sessions import SessionMiddleware

try:
    from ddgs import DDGS  # recherche web gratuite (DuckDuckGo), sans clé
except ImportError:
    DDGS = None

load_dotenv()
E = os.environ.get
BASE = Path(__file__).parent
URL = E("BASE_URL", "http://localhost:8000").rstrip("/")
NTFY = E("NTFY_TOPIC", "").strip()  # ex. https://ntfy.sh/zephon-xxxx : les alertes arrivent sur ton téléphone
TECH = "Un problème technique est survenu. Nous nous en occupons."
SYSTEM = "Tu es Zephon, un assistant IA clair, précis et chaleureux. Réponds dans la langue de l'utilisateur. Si l'utilisateur demande des photos ou des images, utilise l'outil image_search."

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=E("SESSION_SECRET") or secrets.token_hex(32),
                   https_only=URL.startswith("https"), same_site="lax", max_age=60 * 60 * 24 * 30)

db = sqlite3.connect(BASE / "zephon.db", check_same_thread=False)
db.execute("create table if not exists chats(uid text primary key, data text)")

PROV = {  # fournisseurs compatibles OpenAI : (url, nom de la variable de clé)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}
MODELS = {  # ce qui apparaît dans le menu ; le serveur masque les fournisseurs dont la clé manque
    "Gemini": ["gemini-3.8-flash"],
    "Groq": ["openai/gpt-oss-120b"],
}
# Ordre de secours : si un modèle est épuisé (429) ou en panne, on passe au suivant sans rien dire à l'utilisateur
SECOURS = [("gemini", "gemini-3.8-flash"), ("gemini", "gemini-3.6-flash"),
           ("gemini", "gemini-3.5-flash-lite"), ("groq", "openai/gpt-oss-120b")]
HITS = defaultdict(deque)
_sent = {}  # une même alerte n'est envoyée qu'une fois toutes les 10 minutes

async def alert(titre: str, detail: str = ""):
    """Notification sur ton téléphone (ntfy) et dans les logs Render."""
    titre = titre.replace("\n", " ")[:100]
    print(f"[ALERTE] {titre} | {detail}")
    if not NTFY or time.time() - _sent.get(titre, 0) < 600: return
    _sent[titre] = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(NTFY, content=detail[:900].encode("utf-8"),
                         headers={"Title": titre.encode("ascii", "ignore").decode(), "Priority": "high", "Tags": "warning"})
    except Exception:
        pass

class Retry(Exception):
    """Le modèle n'a encore rien envoyé à l'utilisateur : on peut passer au suivant."""

def limit(req: Request, n=20, key="chat"):
    xff = req.headers.get("x-forwarded-for", "")
    # Derrière Render, la vraie adresse du visiteur est la dernière de la liste X-Forwarded-For
    ip = (req.headers.get("cf-connecting-ip") or (xff.split(",")[-1].strip() if xff else "")
          or (req.client.host if req.client else "?"))
    q, now = HITS[(ip, key)], time.time()
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= n: raise HTTPException(429, "Doucement ! Attends une minute avant d'envoyer un nouveau message.")
    q.append(now)

TOOLS = [
    {"type": "function", "function": {"name": "web_search",
        "description": "Cherche sur le web. À utiliser pour l'actualité, les prix, la météo, les scores, les faits récents ou tout ce dont tu n'es pas sûr.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "image_search",
        "description": "Cherche des photos ou images sur le web. À utiliser dès que l'utilisateur demande des images/photos ou veut montrer à quoi ressemble quelque chose.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

def sse(o): return f"data: {json.dumps(o, ensure_ascii=False)}\n\n"

def web(q):
    try: return list(DDGS().text(q, max_results=5)) if DDGS else []
    except Exception: return []

def imgs(q):
    """Recherche d'images (gratuite, DuckDuckGo). Renvoie [{'t', 'u'}] ou []."""
    if not DDGS: return []
    res, vus = [], set()
    mots = q.split()
    essais = (q, " ".join(mots[-4:]), " ".join(mots[:4])) if len(mots) >= 5 else (q,)
    for essai in essais:
        try:
            for x in DDGS().images(f"{essai} photo", max_results=6):
                u = str(x.get("image", ""))
                if u.startswith("https://") and u not in vus:
                    vus.add(u)
                    res.append({"t": x.get("title", "")[:100], "u": u})
            if len(res) >= 4: break
        except Exception:
            continue
    return res[:4]

def sysmsg(extra=""):
    return {"role": "system", "content": f"{SYSTEM} Date du jour : {date.today():%d/%m/%Y}.{extra}"}

async def run_one(prov, model, msgs, st):
    """Une tentative avec un seul modèle. Lève Retry si le modèle refuse avant d'avoir rien envoyé."""
    url, kv = PROV[prov]
    h = {"Authorization": f"Bearer {E(kv)}"}
    can_search = DDGS is not None and "compound" not in model  # compound cherche déjà tout seul
    searched, extra, sources = False, "", []
    try:
        async with httpx.AsyncClient(timeout=None) as c:
            for _ in range(3):
                body = {"model": model, "stream": True, "messages": [sysmsg(extra)] + msgs}
                if can_search and not searched: body["tools"] = TOOLS
                calls = []
                async with c.stream("POST", url, json=body, headers=h) as r:
                    if r.status_code != 200:
                        txt = (await r.aread()).decode(errors="ignore")[:300]
                        if "tools" in body and r.status_code == 400:
                            can_search = False; continue  # ce modèle n'accepte pas les outils
                        quoi = "quota atteint" if r.status_code == 429 else f"erreur {r.status_code}"
                        await alert(f"Zephon - {quoi} ({model})", txt)
                        if not st["sent"]: raise Retry()
                        yield sse({"error": TECH}); return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"): continue
                        d = line[5:].strip()
                        if d == "[DONE]": break
                        try: ch = json.loads(d)["choices"][0]["delta"]
                        except Exception: continue
                        if ch.get("content"):
                            st["sent"] = True
                            yield f"data: {d}\n\n"
                        for tc in ch.get("tool_calls") or []:
                            f = tc.get("function") or {}
                            if f.get("name"):
                                calls.append({"n": f.get("name"), "a": f.get("arguments") or ""})
                            elif calls: calls[-1]["a"] += f.get("arguments") or ""
                if not calls: break
                todo = []
                for call in calls[:2]:
                    try: q = str(json.loads(call["a"] or "{}").get("query", "")).strip()
                    except Exception: q = ""
                    if q: todo.append((call["n"] == "image_search", q))
                if not todo: can_search = False; continue
                for is_img, q in todo:
                    st["sent"] = True
                    if is_img:
                        yield sse({"status": {"type": "images", "q": q}})
                        res = await asyncio.to_thread(imgs, q)
                        yield sse({"images": res})
                        extra += f"\n\nImages trouvées pour « {q} » : " + (
                            ", ".join(x["u"] for x in res) or "(aucune)"
                        ) + "\nMentionne-les brièvement si pertinent, sans recopier les liens."
                    else:
                        yield sse({"status": {"type": "search", "q": q}})
                        res = await asyncio.to_thread(web, q)
                        sources += [{"t": x.get("title", ""), "u": x.get("href", "")} for x in res]
                        extra += f"\n\nRésultats de recherche web pour « {q} » :\n" + (
                            "\n".join(f"- {x.get('title', '')} ({x.get('href', '')}) : {x.get('body', '')}" for x in res) or "(aucun résultat)")
                extra += "\n\nUtilise ces résultats pour répondre, sans inventer ce qu'ils ne disent pas. Dis-le si la recherche n'a rien donné."
                searched = True
            if sources: yield sse({"sources": sources[:8]})
            yield "data: [DONE]\n\n"
    except httpx.HTTPError as e:
        await alert("Zephon - connexion IA impossible", f"{model} : {e!r}"[:300])
        if not st["sent"]: raise Retry()
        yield sse({"error": TECH})

async def run(chain, msgs):
    """Essaie chaque modèle de la chaîne jusqu'à ce qu'un réponde."""
    for i, (prov, model) in enumerate(chain):
        st = {"sent": False}
        try:
            async for chunk in run_one(prov, model, msgs, st):
                yield chunk
            return
        except Retry:
            if i + 1 < len(chain):
                await alert(f"Zephon - bascule depuis {model}", f"{model} indisponible, passage à {chain[i + 1][1]}.")
                continue
            await alert("Zephon - aucun modèle disponible", "Tous les modèles de secours ont échoué.")
            yield sse({"error": TECH})
            return

@app.post("/api/chat")
async def chat(req: Request):
    limit(req)
    b = await req.json()
    prov, _, model = str(b.get("model", "")).partition(":")
    if prov not in PROV or not model:
        await alert("Zephon - modèle inconnu", str(b.get("model", ""))[:100])
        raise HTTPException(400, TECH)
    msgs = [{"role": m["role"], "content": str(m["content"])[:20000]}
            for m in b.get("messages", [])[-40:] if m.get("role") in ("user", "assistant")]
    chain = [(prov, model)] + [x for x in SECOURS if x != (prov, model)]
    chain = [(p, m) for p, m in chain if E(PROV[p][1])]  # on ne garde que les fournisseurs qui ont une clé
    if not chain:
        await alert("Zephon - aucune cle IA", "Aucune clé API trouvée sur le serveur (Render > Environment).")
        raise HTTPException(500, TECH)
    return StreamingResponse(run(chain, msgs), media_type="text/event-stream")

@app.post("/api/report")
async def report(req: Request):
    """Bug repéré dans le navigateur : on te l'envoie sur ton téléphone."""
    limit(req, n=5, key="report")
    try: b = await req.json()
    except Exception: b = {}
    msg = str(b.get("msg", ""))[:500]
    await alert(f"Zephon - bug : {msg[:50]}", f"{msg}\n{str(b.get('where', ''))[:200]}")
    return {"ok": True}

_cache = {"t": 0, "v": None}

async def live(prov):  # liste réelle des modèles du fournisseur (None si injoignable)
    url = PROV[prov][0].rsplit("/chat/completions", 1)[0] + "/models"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(url, headers={"Authorization": f"Bearer {E(PROV[prov][1])}"})
        return {m["id"].removeprefix("models/") for m in r.json()["data"]}
    except Exception:
        return None

@app.get("/api/models")
async def models():
    if _cache["v"] is not None and time.time() - _cache["t"] < 3600: return _cache["v"]
    out = {}
    for name, ids in MODELS.items():
        prov = name.lower()
        if not E(PROV[prov][1]): continue  # pas de clé = fournisseur masqué
        have = await live(prov)
        out[name] = [m for m in ids if have is None or m in have] or ids
    _cache.update(t=time.time(), v=out)
    return out

# ---------- connexion Google / GitHub ----------
OA = {
    "google": dict(auth="https://accounts.google.com/o/oauth2/v2/auth", token="https://oauth2.googleapis.com/token",
                   info="https://openidconnect.googleapis.com/v1/userinfo", scope="openid email profile",
                   id=E("GOOGLE_CLIENT_ID"), secret=E("GOOGLE_CLIENT_SECRET")),
    "github": dict(auth="https://github.com/login/oauth/authorize", token="https://github.com/login/oauth/access_token",
                   info="https://api.github.com/user", scope="read:user",
                   id=E("GITHUB_CLIENT_ID"), secret=E("GITHUB_CLIENT_SECRET")),
}

@app.get("/auth/{p}")
async def auth(p: str, req: Request):
    if p == "logout":
        req.session.clear(); return RedirectResponse("/")
    o = OA.get(p)
    if not o: raise HTTPException(404)
    if not o["id"]:
        await alert(f"Zephon - connexion {p} non configuree", f"Variable {p.upper()}_CLIENT_ID absente sur Render.")
        raise HTTPException(503, TECH)
    st = secrets.token_urlsafe(16); req.session["st"] = st
    q = httpx.QueryParams({"client_id": o["id"], "redirect_uri": f"{URL}/auth/{p}/callback",
                           "response_type": "code", "scope": o["scope"], "state": st})
    return RedirectResponse(f'{o["auth"]}?{q}')

@app.get("/auth/{p}/callback")
async def callback(p: str, req: Request, code: str = "", state: str = ""):
    o = OA.get(p)
    if not o or not code or state != req.session.pop("st", None): raise HTTPException(400, "Connexion refusée.")
    async with httpx.AsyncClient() as c:
        t = (await c.post(o["token"], headers={"Accept": "application/json"}, data={
            "client_id": o["id"], "client_secret": o["secret"], "code": code,
            "redirect_uri": f"{URL}/auth/{p}/callback", "grant_type": "authorization_code"})).json()
        u = (await c.get(o["info"], headers={"Authorization": f'Bearer {t.get("access_token")}'})).json()
    if p == "google" and "sub" in u:
        user = {"id": "g:" + u["sub"], "name": u.get("given_name") or u.get("name", ""), "photo": u.get("picture", "")}
    elif p == "github" and "id" in u:
        user = {"id": "h:" + str(u["id"]), "name": (u.get("name") or u.get("login", "")).split(" ")[0], "photo": u.get("avatar_url", "")}
    else: raise HTTPException(400, "Connexion refusée.")
    req.session["u"] = user
    return RedirectResponse("/")

def me(req: Request):
    u = req.session.get("u")
    if not u: raise HTTPException(401, "Non connecté.")
    return u

@app.get("/api/me")
def api_me(req: Request): return me(req)

@app.get("/api/chats")
def get_chats(req: Request):
    r = db.execute("select data from chats where uid=?", (me(req)["id"],)).fetchone()
    return json.loads(r[0]) if r else []

@app.put("/api/chats")
async def put_chats(req: Request):
    u, d = me(req), await req.json()
    s = json.dumps(d[:100]) if isinstance(d, list) else ""
    if not s or len(s) > 5_000_000: raise HTTPException(400, "Données invalides.")
    db.execute("insert or replace into chats values(?,?)", (u["id"], s)); db.commit()
    return {"ok": True}

# ---------- pages ----------
@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

@app.exception_handler(SHTTP)
async def errors(req: Request, exc: SHTTP):
    if exc.status_code == 404 and not req.url.path.startswith("/api"):
        return RedirectResponse("/")  # page inconnue : retour à l'accueil
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def crash(req: Request, exc: Exception):
    """Bug imprévu : on te prévient, et le visiteur voit un message simple."""
    await alert("Zephon - plantage serveur", f"{req.method} {req.url.path} : {exc!r}"[:400])
    return JSONResponse({"detail": TECH}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=E("HOST", "127.0.0.1"), port=int(E("PORT", "8000")))
