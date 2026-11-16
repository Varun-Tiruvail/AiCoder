"""NB Studio: local web UI. Writes notebook cells with GPT-4o. Never executes code."""
import os, re, ast, json, subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from openai import OpenAI
import sqlglot
from sqlglot import exp

APP = Path(__file__).parent
ROOT = Path(os.environ.get("PROJECT_DIR", APP / "project")).resolve()
NB, MEM, TASKS = ROOT / "work.ipynb", ROOT / "memory", ROOT / "tasks.json"
CFG = json.loads((APP / "config.json").read_text())
ROOT.mkdir(exist_ok=True); MEM.mkdir(exist_ok=True)
if not (MEM / "notes.md").exists():
    (MEM / "notes.md").write_text("# Project notes\nWrite goals, data schema, conventions here.\n")
app = Flask(__name__, static_folder="static")
client = OpenAI()  # reads OPENAI_API_KEY

# ---------- notebook ----------
def load_nb():
    return nbformat.read(NB, 4) if NB.exists() else new_notebook()
def save_nb(nb): nbformat.write(nb, NB)
def cells(nb): return [{"i": i + 1, "source": c.source} for i, c in enumerate(nb.cells)]
def summary(nb):
    """Cheap context: only names defined so far, not full code."""
    out = []
    for i, c in enumerate(nb.cells, 1):
        b = sql_body(c.source)
        if b is not None:
            tabs = []
            try:
                for s in sqlglot.parse(b, read=CFG.get("dialect")):
                    if isinstance(s, exp.Create) and s.find(exp.Table):
                        cols = s.expression.named_selects if s.expression else []
                        tabs.append(f"{s.find(exp.Table).name}({', '.join(cols)})")
            except Exception: pass
            out.append(f"cell {i}: sql, creates: {'; '.join(tabs) or '-'}"); continue
        try: tree = ast.parse(c.source)
        except SyntaxError: continue
        names = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)): names.append(n.name)
            elif isinstance(n, ast.Assign):
                names += [t.id for t in n.targets if isinstance(t, ast.Name)]
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names += [(a.asname or a.name) for a in n.names]
        out.append(f"cell {i}: {', '.join(names) or '-'}")
    return "\n".join(out)

# ---------- whitelist (AST only, nothing is run) ----------
def dotted(n):
    if isinstance(n, ast.Name): return n.id
    if isinstance(n, ast.Attribute):
        b = dotted(n.value); return f"{b}.{n.attr}" if b else n.attr
    return None
def sql_body(code):
    lines = code.strip().splitlines()
    return "\n".join(lines[1:]) if lines and lines[0].lstrip().startswith("%%sql") else None

def check_sql(sql):
    try: stmts = [s for s in sqlglot.parse(sql, read=CFG.get("dialect")) if s]
    except Exception as e: return [f"SQL parse error: {e}"]
    okst = set(CFG["allowed_statements"])
    okfn = {f.lower() for f in CFG["allowed_sql_functions"]}
    prefix, errs = CFG.get("temp_prefix", ""), []
    for s in stmts:
        kind = type(s).__name__
        if kind not in okst: errs.append(f"Statement not allowed: {kind.upper()}"); continue
        if isinstance(s, exp.Create):
            t = s.find(exp.Table)
            if str(s.args.get("kind")).upper() != "TABLE" or not s.find(exp.TemporaryProperty):
                errs.append("Only CREATE TEMP TABLE is allowed")
            elif prefix and not t.name.startswith(prefix):
                errs.append(f"Temp table name must start with {prefix}")
        for f in s.find_all(exp.Func):
            n = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
            if n not in okfn: errs.append(f"Function not allowed: {n}")
    return list(dict.fromkeys(errs))

def check(code):
    body = sql_body(code)
    if body is not None: return check_sql(body)
    if CFG.get("sql_only"): return ["Cell must start with %%sql"]
    try: tree = ast.parse(code)
    except SyntaxError as e: return [f"SyntaxError: {e}"]
    local = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    ok, errs = set(CFG["allowed_functions"]), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            d = dotted(n.func) or (n.func.attr if isinstance(n.func, ast.Attribute) else "?")
            if d not in ok and d.split(".")[-1] not in ok and d not in local:
                errs.append(f"Call not allowed: {d}")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [n.module] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names]
            errs += [f"Import not allowed: {m}" for m in mods if m not in CFG["allowed_imports"]]
    return errs

# ---------- tasks / memory ----------
def tasks(): return json.loads(TASKS.read_text()) if TASKS.exists() else []
def memtext(): return (MEM / "notes.md").read_text()[:4000]

def ask(system, user):
    r = client.chat.completions.create(model=CFG["model"], temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)

@app.get("/")
def home(): return send_from_directory("static", "index.html")

@app.get("/api/state")
def state():
    return jsonify(cells=cells(load_nb()), tasks=tasks(),
                   memory=sorted(p.name for p in MEM.glob("*") if p.is_file()))

@app.post("/api/plan")
def plan():
    goal = request.json["goal"]
    res = ask('Return JSON {"tasks": [short strings]}. Break the goal into 4-10 small, ordered notebook tasks, '
              'each doable in one code cell.', f"Notes:\n{memtext()}\n\nGoal: {goal}")
    t = [{"text": x, "done": False} for x in res["tasks"]]
    TASKS.write_text(json.dumps(t)); return jsonify(tasks=t)

@app.put("/api/tasks")
def put_tasks(): TASKS.write_text(json.dumps(request.json)); return jsonify(ok=True)

@app.put("/api/cell/<int:i>")
def put_cell(i):
    nb = load_nb(); nb.cells[i - 1].source = request.json["source"]; save_nb(nb); return jsonify(ok=True)
@app.delete("/api/cell/<int:i>")
def del_cell(i):
    nb = load_nb(); nb.cells.pop(i - 1); save_nb(nb); return jsonify(ok=True)

@app.post("/api/chat")
def chat():
    msg = request.json["message"]
    nb = load_nb()
    mc, mt = re.search(r"\bcell\s*#?(\d+)", msg, re.I), re.search(r"\btask\s*#?(\d+)", msg, re.I)
    tl = tasks()
    ci = int(mc.group(1)) if mc else None
    if ci and not 1 <= ci <= len(nb.cells): return jsonify(reply=f"Cell {ci} does not exist."), 400
    task = None
    if mt and 1 <= int(mt.group(1)) <= len(tl): task = tl[int(mt.group(1)) - 1]["text"]
    elif not ci: task = next((t["text"] for t in tl if not t["done"]), None)
    if CFG.get("sql_only"):
        system = ('You write ONE Jupyter cell. Return JSON {"code": str, "note": str}. The code MUST start with the line %%sql '
                  "and contain only SQL. Nothing is executed. Allowed statements: " + ", ".join(CFG["allowed_statements"]) +
                  ". Temp tables only: CREATE TEMP TABLE " + CFG.get("temp_prefix", "") + "name AS SELECT ... Never DROP or alter. "
                  "Allowed functions: " + ", ".join(CFG["allowed_sql_functions"]) + ". Do not rewrite other cells. Keep it short.")
    else: system = ('You write ONE Jupyter code cell. Return JSON {"code": str, "note": str}. '
              "Code is never executed. Use ONLY these functions: " + ", ".join(CFG["allowed_functions"]) +
              ". Only these imports: " + ", ".join(CFG["allowed_imports"]) +
              ". Do not rewrite other cells. Keep it short.")
    user = (f"Notes:\n{memtext()}\n\nDefined so far:\n{summary(nb) or '-'}\n\n"
            + (f"Task: {task}\n" if task else "")
            + (f"Existing code of cell {ci}:\n{nb.cells[ci-1].source}\n" if ci else "")
            + f"Request: {msg}")
    for _ in range(3):
        res = ask(system, user); errs = check(res["code"])
        if not errs: break
        user += f"\n\nYour last code was rejected: {errs}. Fix it:\n{res['code']}"
    else:
        return jsonify(reply="Rejected after retries: " + "; ".join(errs)), 422
    if ci: nb.cells[ci - 1].source = res["code"]
    else: nb.cells.append(new_code_cell(res["code"])); ci = len(nb.cells)
    save_nb(nb)
    return jsonify(reply=f"Cell {ci}: {res.get('note', 'done')}", cell=ci, cells=cells(nb))

# ---------- memory ----------
@app.route("/api/memory/<name>", methods=["GET", "PUT"])
def memory(name):
    p = MEM / Path(name).name
    if request.method == "PUT": p.write_text(request.json["text"]); return jsonify(ok=True)
    return jsonify(text=p.read_text() if p.exists() else "")

# ---------- git ----------
GIT = {"status": ["status", "-sb"], "pull": ["pull"], "push": ["push"], "log": ["log", "--oneline", "-10"],
       "branches": ["branch", "-a"], "diff": ["diff", "--stat"], "init": ["init"]}
@app.post("/api/git")
def git():
    cmd, arg = request.json["cmd"], request.json.get("arg", "").strip()
    if arg.startswith("-"): return jsonify(out="Invalid argument"), 400
    if cmd == "commit": args = ["commit", "-am", arg or "update"]; subprocess.run(["git", "add", "-A"], cwd=ROOT)
    elif cmd == "merge" and arg: args = ["merge", arg]
    elif cmd == "checkout" and arg: args = ["checkout", arg]
    elif cmd == "newbranch" and arg: args = ["checkout", "-b", arg]
    elif cmd in GIT: args = GIT[cmd]
    else: return jsonify(out="Unknown command"), 400
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    return jsonify(out=(r.stdout + r.stderr).strip() or "ok")

if __name__ == "__main__":
    app.run(port=5000, debug=False)
