 
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import timedelta
from functools import wraps
from urllib.parse import urlparse

from flask import Flask, abort, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from data.careers import CAREERS
from data.i18n import T
from data.labels import AREAS, BUDGET, CATS, EDU, GOALS, SKILLS, STATES, STREAMS
from recommender import Recommender

app = Flask(__name__)
os.makedirs(app.instance_path, exist_ok=True)


def _secret_key():
    """SECRET_KEY env var if set, else a random key saved once in instance/secret_key."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(app.instance_path, "secret_key")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key


app.config.update(SECRET_KEY=_secret_key(), SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=30))

recommender = Recommender()          # loads (or trains) the ML model once at start-up

CAT = {c["id"]: c for c in CATS}
STREAM_IDS = {s["id"] for s in STREAMS}
SKILL_IDS = {s["id"] for s in SKILLS}
GOAL_IDS = {g["v"] for g in GOALS}
AREA_IDS = {a["v"] for a in AREAS}
BUDGET_IDS = {b["v"] for b in BUDGET}
EDU_VALUES = {float(e["v"]) for e in EDU}


# ------------------------------------------------------------------ database (SQLite)
def db():
    if "db" not in g:
        g.db = sqlite3.connect(os.path.join(app.instance_path, "disha.db"))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


with app.app_context():
    db().execute("""CREATE TABLE IF NOT EXISTS users(
        mobile TEXT PRIMARY KEY, name TEXT NOT NULL, pw_hash TEXT NOT NULL, profile TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    db().commit()


# ------------------------------------------------------------------ language helpers
def lang():
    return session.get("lang") if session.get("lang") in ("en", "hi") else "en"


def t(key):
    return T[lang()].get(key) or T["en"].get(key, key)


def L(obj):
    return obj.get(lang()) or obj["en"]


def skill_name(sid):
    s = next((x for x in SKILLS if x["id"] == sid), None)
    return L(s) if s else sid


def edu_label(v):
    if v == 0:
        return t("anyEdu")
    return L(next((e for e in EDU if e["v"] == v), EDU[0]))


def cost_label(v):
    return L(next((b for b in BUDGET if b["v"] == v), BUDGET[0]))


# ------------------------------------------------------------------ CSRF + session helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


@app.before_request
def csrf_protect():
    if request.method == "POST":
        sent = request.form.get("_csrf", "").encode()
        if not hmac.compare_digest(sent, session.get("_csrf", "").encode()):
            abort(400)


@app.after_request
def no_cache(resp):
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store"
    return resp


def current_user():
    return session.get("mobile") or session.get("guest")


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


@app.context_processor
def inject():
    return dict(t=t, L=L, lang=lang(), cat_of=CAT.get, skill_name=skill_name, edu_label=edu_label,
                cost_label=cost_label, csrf_token=csrf_token, user_name=session.get("name", ""),
                careers_count=len(CAREERS), EDU=EDU, STREAMS=STREAMS, SKILLS=SKILLS, CATS=CATS,
                GOALS=GOALS, BUDGET=BUDGET, AREAS=AREAS, STATES=STATES)


def blank_profile(name=""):
    return dict(name=name, age="", gender="", state="", area="village", edu=None, stream="",
                skills=[], interests=[], goal="", internet="yes", relocate="yes", budget=1)


def start_session(mobile, name, profile_json):
    keep = session.get("lang")
    session.clear()                                    # new session id on login
    if keep:
        session["lang"] = keep
    session.permanent = True
    session["name"] = name
    if mobile:
        session["mobile"] = mobile
    else:
        session["guest"] = True
    if profile_json:
        session["profile"] = json.loads(profile_json)
        session["done"] = True
    else:
        session["profile"] = blank_profile(name)


# ------------------------------------------------------------------ login / signup / guest
def auth(mode):
    if current_user():
        return redirect(url_for("home"))
    error, form = "", dict(name="", mobile="")
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:60]
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "")
        form.update(name=name, mobile=mobile)
        if not re.fullmatch(r"[6-9][0-9]{9}", mobile):
            error = t("errMobile")
        elif len(password) < 6:
            error = t("errPass")
        elif mode == "signup" and not name:
            error = t("errName")
        else:
            row = db().execute("SELECT * FROM users WHERE mobile=?", (mobile,)).fetchone()
            if mode == "signup":
                if row:
                    error = t("errExists")
                else:
                    db().execute("INSERT INTO users(mobile,name,pw_hash) VALUES(?,?,?)",
                                 (mobile, name, generate_password_hash(password)))
                    db().commit()
                    start_session(mobile, name, None)
                    return redirect(url_for("home"))
            elif not row or not check_password_hash(row["pw_hash"], password):
                error = t("errCred")
            else:
                start_session(mobile, row["name"], row["profile"])
                return redirect(url_for("home"))
    return render_template("auth.html", mode=mode, error=error, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    return auth("login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    return auth("signup")


@app.route("/guest", methods=["POST"])
def guest():
    start_session(None, "मेहमान" if lang() == "hi" else "Guest", None)
    return redirect(url_for("home"))


@app.route("/logout", methods=["POST"])
def logout():
    keep = session.get("lang")
    session.clear()
    if keep:
        session["lang"] = keep
    return redirect(url_for("login"))


@app.route("/lang/<code>")
def set_lang(code):
    if code in ("en", "hi"):
        session["lang"] = code
    ref = urlparse(request.referrer or "")
    if ref.netloc == request.host:                     # go back to the page the user was on
        return redirect(ref.path + ("?" + ref.query if ref.query else ""))
    return redirect(url_for("home"))


@app.route("/")
def home():
    if not current_user():
        return redirect(url_for("login"))
    return redirect(url_for("results") if session.get("done") else url_for("wizard", step=0))


# ------------------------------------------------------------------ 3-step profile form
def parse_step(step, f, p):
    if step == 0:
        p["name"] = f.get("name", "").strip()[:60]
        p["age"] = f.get("age", "").strip()[:3]
        p["gender"] = f.get("gender", "") if f.get("gender", "") in ("", "m", "f", "o") else ""
        p["state"] = f.get("state", "") if f.get("state", "") in STATES else ""
        p["area"] = f.get("area", "") if f.get("area", "") in AREA_IDS else "village"
    elif step == 1:
        try:
            edu = float(f.get("edu", ""))
        except ValueError:
            edu = None
        p["edu"] = edu if edu in EDU_VALUES else None
        stream = f.get("stream", "")
        p["stream"] = stream if stream in STREAM_IDS and (p["edu"] or 0) >= 3 else ""
        p["skills"] = [s for s in f.getlist("skills") if s in SKILL_IDS]
    else:
        p["interests"] = [c for c in f.getlist("interests") if c in CAT]
        p["goal"] = f.get("goal", "") if f.get("goal", "") in GOAL_IDS else ""
        p["internet"] = "no" if f.get("internet") == "no" else "yes"
        p["relocate"] = "no" if f.get("relocate") == "no" else "yes"
        try:
            b = int(f.get("budget", "1"))
        except ValueError:
            b = 1
        p["budget"] = b if b in BUDGET_IDS else 1


def validate_step(step, p):
    if step == 0:
        try:
            age = int(p["age"])
        except ValueError:
            return "eState"
        if not 14 <= age <= 45 or not p["state"]:
            return "eState"
    if step == 1 and (p["edu"] is None or not p["skills"]):
        return "eEdu"
    if step == 2 and (not p["interests"] or not p["goal"]):
        return "eInt"
    return ""


@app.route("/profile/<int:step>", methods=["GET", "POST"])
@login_required
def wizard(step):
    if step not in (0, 1, 2):
        abort(404)
    p = session.get("profile") or blank_profile(session.get("name", ""))
    for earlier in range(step):                        # cannot skip ahead of an unfinished step
        if validate_step(earlier, p):
            return redirect(url_for("wizard", step=earlier))
    error = ""
    if request.method == "POST":
        parse_step(step, request.form, p)
        session["profile"] = p
        if "back" in request.form:                     # keep what was typed, no validation
            return redirect(url_for("wizard", step=max(0, step - 1)))
        error = t(validate_step(step, p)) if validate_step(step, p) else ""
        if not error:
            if step < 2:
                return redirect(url_for("wizard", step=step + 1))
            if not p["name"]:
                p["name"] = session.get("name", "")
            session["profile"], session["done"] = p, True
            if session.get("mobile"):
                db().execute("UPDATE users SET profile=? WHERE mobile=?", (json.dumps(p), session["mobile"]))
                db().commit()
            return redirect(url_for("analyzing"))
    names = [t("s1"), t("s2"), t("s3")]
    return render_template("wizard.html", step=step, p=p, error=error, names=names, active="path")


@app.route("/analyzing")
@login_required
def analyzing():
    if not session.get("done"):
        return redirect(url_for("wizard", step=0))
    return render_template("analyzing.html", active="path")


@app.route("/results")
@login_required
def results():
    p = session.get("profile")
    if not session.get("done") or not p:
        return redirect(url_for("wizard", step=0))
    R = recommender.recommend(p, t, skill_name, edu_label)
    edu = next((e for e in EDU if e["v"] == p["edu"]), None)
    summary = [L(edu) if edu else "", p["state"]] + [skill_name(k) for k in p["skills"][:4]]
    return render_template("results.html", R=R, p=p, summary=[x for x in summary if x], active="path")


@app.route("/explore")
@login_required
def explore():
    cat = request.args.get("cat", "all")
    if cat != "all" and cat not in CAT:
        cat = "all"
    q = request.args.get("q", "").strip()[:60]
    ql = q.lower()
    items = [c for c in CAREERS
             if (cat == "all" or c["cat"] == cat) and (not ql or ql in (c["en"] + " " + c["hi"] + " " + c["desc"]).lower())]
    return render_template("explore.html", items=items, cat=cat, q=q, active="explore")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
