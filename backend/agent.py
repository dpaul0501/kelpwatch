"""The KelpWatch agent: a tool-calling loop with policy checks, verification and a trace.

Flow for every question:
  1. Input policy   - scope and personal-data rules run in code before any model call
  2. Agent loop     - the model plans, calls approved tools, and reads their results
  3. Verification   - every figure in the answer is matched against tool results
  4. Output policy  - evidence, citation, disclosure and personal-data rules
The trace records each step so the UI can show exactly what happened.
"""
import json, pathlib, re, time

from policy import PolicyEngine
from verify import verify_answer

SKILLS_DIR = pathlib.Path(__file__).resolve().parent / "skills"
MAX_TURNS = 6
MAX_TOOL_CHARS = 6000          # context budget per tool result

TOOL_SPECS = [
    {"name": "get_kelp_trends",
     "description": "Nearshore vegetation signal from Landsat for six Puget Sound counties across four periods, with quality checks and method caveats. Optionally filter to one county.",
     "parameters": {"type": "object", "properties": {"county": {"type": ["string", "null"], "description": "King, Skagit, Whatcom, Kitsap, Pierce or Snohomish"}}}},
    {"name": "get_wildfire_activity",
     "description": "Active fire detections (NASA FIRMS via Earth Engine) for the last 7 days over the US and India.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_drought_conditions",
     "description": "India rainfall over the last 30 days compared with the 1991-2020 normal (CHIRPS), and vegetation health (MODIS NDVI).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search_documents",
     "description": "Search the knowledge base of public kelp and eelgrass documents (WA DNR, WDFW, Northwest Straits, NOAA partners). Returns numbered passages [D1], [D2]... with title and page.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "rank_restoration_sites",
     "description": "Rank an ILLUSTRATIVE list of restoration sites by cost per acre and habitat benefit. Not official ESRP data.",
     "parameters": {"type": "object", "properties": {"budget_usd": {"type": ["number", "null"]}}}},
    {"name": "get_data_quality",
     "description": "Status and quality checks for every data source and the document knowledge base.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "use_skill",
     "description": "Load a skill: a step-by-step procedure for a common task. Call this first when a skill matches the question.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]


def load_skills() -> dict:
    skills = {}
    for p in sorted(SKILLS_DIR.glob("*.md")):
        text = p.read_text()
        m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
        meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
        skills[meta["name"]] = {"name": meta["name"], "description": meta["description"], "body": m.group(2).strip()}
    return skills


SYSTEM_PROMPT = """You are KelpWatch, a climate data agent for Puget Sound kelp and eelgrass, wildfire, and drought risk.

How to work:
- Call tools to get data. Never answer from memory when a tool can provide the data.
- When you need several tools, call them together in one step rather than one at a time.
- If a skill matches the question, call use_skill first and follow its steps.
- Use only figures that appear in tool results. Do not calculate new figures unless you show the inputs.
- When you use documents, base each claim on a retrieved passage and cite it; do not add background knowledge the passages don't contain.
- Cite document passages as [D1], [D2] using the numbers search_documents gives you, in plain square brackets. Only cite passages you actually retrieved.
- Copy figures exactly as the tools give them (rounding to one decimal place is fine). Say which county, period or region each figure belongs to.
- If a tool result reports a failed quality check or a caveat, say so plainly.
- The restoration site list is illustrative; say so whenever you use it.
- Be concise: under 220 words, short paragraphs or bullets. Do not use tables.

Available skills:
{skills}"""


class Agent:
    def __init__(self, tools: dict, client, model: str, kb, reasoning_effort: str | None = None):
        self.reasoning_effort = reasoning_effort
        self.tools = tools          # name -> callable(**args) returning dict
        self.client = client
        self.model = model
        self.kb = kb
        self.policy = PolicyEngine()
        self.skills = load_skills()

    def _run_tool(self, name: str, args: dict, doc_sources: list):
        if name == "use_skill":
            s = self.skills.get(args.get("name", ""))
            return {"skill": s["name"], "procedure": s["body"]} if s else {"error": f"No skill named {args.get('name')}. Available: {', '.join(self.skills)}"}
        if name == "search_documents":
            hits = self.kb.search(args.get("query", ""), k=3)
            out = []
            for h in hits:
                doc_sources.append(h)
                out.append({"ref": f"D{len(doc_sources)}", "title": h["title"], "publisher": h["publisher"],
                            "year": h["year"], "page": h["page"], "text": h["text"][:650]})
            return {"passages": out, "method": "BM25 keyword retrieval over ingested public documents"}
        return self.tools[name](**args)

    def run(self, question: str) -> dict:
        trace, t0 = [], time.time()
        q, input_results, refused = self.policy.check_input(question)
        trace.append({"type": "policy", "stage": "input", "results": input_results})
        if refused:
            return {"answer": "I can only help with kelp and eelgrass, wildfire, and drought and monsoon risk, "
                              "and the data and documents behind them. Try asking about one of those.",
                    "refused": True, "trace": trace, "policy": input_results, "verification": None,
                    "citations": [], "model": None, "elapsed_ms": int((time.time() - t0) * 1000)}

        skills_list = "\n".join(f"- {s['name']}: {s['description']}" for s in self.skills.values())
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(skills=skills_list)},
                    {"role": "user", "content": q}]
        tools = [{"type": "function", "function": t} for t in TOOL_SPECS]
        tool_results, tools_used, doc_sources, quality_issues = [], [], [], []
        answer, tool_policy = "", []

        for turn in range(1, MAX_TURNS + 1):
            ts = time.time()
            try:
                extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
                resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                           tool_choice="auto", temperature=0.1, max_tokens=1200,
                                                           extra_body=extra)
            except Exception as e:  # model or provider error: report it, don't crash
                trace.append({"type": "model", "turn": turn, "ms": int((time.time() - ts) * 1000), "tool_calls": [],
                              "error": f"{type(e).__name__}: {str(e)[:300]}"})
                if turn < MAX_TURNS and "tool_use_failed" in str(e):
                    continue          # the model produced a malformed tool call; let it try again
                if turn < MAX_TURNS and ("rate_limit" in str(e) or "429" in str(e)) and "quota" not in str(e).lower():
                    time.sleep(4)     # provider rate limit: wait briefly and retry once more
                    continue
                answer = "The language model returned an error, so I can't answer right now. Please try again."
                break
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            trace.append({"type": "model", "turn": turn, "ms": int((time.time() - ts) * 1000),
                          "tool_calls": [c.function.name for c in (msg.tool_calls or [])],
                          "prompt_tokens": getattr(usage, "prompt_tokens", None),
                          "completion_tokens": getattr(usage, "completion_tokens", None)})
            if not msg.tool_calls:
                answer = msg.content or ""
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in msg.tool_calls]})
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = {k: v for k, v in (json.loads(call.function.arguments or "{}") or {}).items() if v is not None}
                except json.JSONDecodeError:
                    args = {}
                check = self.policy.check_tool(name)
                tool_policy.append(check)
                tt = time.time()
                if not check["passed"]:
                    result = {"error": f"Tool {name} is not approved by policy."}
                else:
                    try:
                        result = self._run_tool(name, args, doc_sources)
                    except Exception as e:  # a failing tool is reported, never hidden
                        result = {"error": f"{type(e).__name__}: {e}"}
                tools_used.append(name)
                tool_results.append(result)
                for issue in result.get("quality_issues", []) if isinstance(result, dict) else []:
                    if issue not in quality_issues:
                        quality_issues.append(issue)
                if isinstance(result, dict) and result.get("error"):
                    quality_issues.append(f"{name} failed: {result['error']}")
                payload = json.dumps(result, default=str)
                truncated = len(payload) > MAX_TOOL_CHARS
                trace.append({"type": "tool", "name": name, "args": args, "allowed": check["passed"],
                              "ms": int((time.time() - tt) * 1000), "chars": len(payload), "truncated": truncated,
                              "summary": _summarise(name, result)})
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": payload[:MAX_TOOL_CHARS] + (" …[truncated]" if truncated else "")})
        else:
            answer = answer or "I couldn't finish within the step limit. Please ask a narrower question."

        # Normalise citation styles some models use (【D2】, 【D2†L1】) and drop citations of tool names.
        answer = re.sub(r"【\s*(D\d+)[^】]*】", r"[\1]", answer)
        answer = re.sub(r"【[^】]*】", "", answer)
        verification = verify_answer(answer, tool_results, q)
        trace.append({"type": "verification", **{k: verification[k] for k in
                      ("figures_checked", "figures_verified", "share_verified", "unverified")}})
        answer, output_results = self.policy.check_output(answer, {
            "verification": verification, "tools_used": tools_used,
            "doc_sources": doc_sources, "quality_issues": quality_issues})
        trace.append({"type": "policy", "stage": "output", "results": output_results})

        cited = sorted({int(n) for n in re.findall(r"\[D(\d+)\]", answer)})
        citations = [{"ref": f"D{n}", "title": doc_sources[n - 1]["title"], "page": doc_sources[n - 1]["page"],
                      "url": doc_sources[n - 1]["url"]} for n in cited if 0 < n <= len(doc_sources)]
        return {"answer": answer, "refused": False, "trace": trace,
                "policy": input_results + tool_policy + output_results,
                "verification": verification, "citations": citations, "tools_used": tools_used,
                "model": self.model, "elapsed_ms": int((time.time() - t0) * 1000)}


def _summarise(name: str, r) -> str:
    if not isinstance(r, dict):
        return ""
    if r.get("error"):
        return f"error: {r['error']}"
    if name == "search_documents":
        return "; ".join(f"{p['ref']} {p['title'][:48]} p.{p['page']}" for p in r.get("passages", []))
    if name == "use_skill":
        return f"loaded skill '{r.get('skill')}'"
    if name == "get_kelp_trends":
        return f"{len(r.get('counties', []))} counties · {r.get('status', '')}"
    if name == "rank_restoration_sites":
        return f"{len(r.get('sites', []))} illustrative sites ranked"
    keys = [k for k in r.keys() if k not in ("method", "caveats", "quality_issues")]
    return ", ".join(keys[:6])
