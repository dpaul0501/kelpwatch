"""Policy engine: loads policy.yaml and runs each rule's check at its stage.

Checks are deterministic code. Each result records the rule, whether it passed,
and the action taken, so every answer carries its own policy record.
"""
import pathlib, re
import yaml

POLICY_PATH = pathlib.Path(__file__).resolve().parent / "policy.yaml"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")


def load_policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text())


def _result(rule, passed, action=None, detail=""):
    return {"rule": rule["id"], "text": rule["text"], "stage": rule["stage"],
            "passed": passed, "action": None if passed else (action or rule["on_fail"]), "detail": detail}


def _redact(text: str) -> tuple[str, int]:
    n = len(EMAIL.findall(text)) + len(PHONE.findall(text))
    return PHONE.sub("[phone redacted]", EMAIL.sub("[email redacted]", text)), n


class PolicyEngine:
    def __init__(self):
        self.policy = load_policy()
        self.rules = self.policy["rules"]

    def by_stage(self, stage):
        return [r for r in self.rules if r["stage"] == stage]

    # ── input ──────────────────────────────────────────────────────────────
    def check_input(self, question: str) -> tuple[str, list, bool]:
        """Returns (possibly redacted question, results, refused)."""
        results, refused = [], False
        for rule in self.by_stage("input"):
            if rule["check"] == "scope":
                q = question.lower()
                hit = next((w for w in rule["params"]["out_of_scope"] if w in q), None)
                results.append(_result(rule, hit is None, detail=f"matched '{hit}'" if hit else "in scope"))
                refused = refused or hit is not None
            elif rule["check"] == "pii":
                question, n = _redact(question)
                results.append(_result(rule, n == 0, detail=f"{n} item(s) redacted" if n else "none found"))
        return question, results, refused

    # ── tool calls ─────────────────────────────────────────────────────────
    def check_tool(self, name: str) -> dict:
        rule = next(r for r in self.by_stage("tool") if r["check"] == "tool_allowlist")
        ok = name in rule["params"]["allowed"]
        return _result(rule, ok, detail=f"{name} {'allowed' if ok else 'is not an approved tool'}")

    # ── output ─────────────────────────────────────────────────────────────
    def check_output(self, answer: str, ctx: dict) -> tuple[str, list]:
        """ctx: verification, tools_used, doc_sources, quality_issues."""
        results = []
        for rule in self.by_stage("output"):
            c = rule["check"]
            if c == "figures_traceable":
                v = ctx["verification"]
                ok = v["share_verified"] >= rule["params"]["min_share"]
                results.append(_result(rule, ok, detail=f"{v['figures_verified']} of {v['figures_checked']} figures traced to data"
                                       + (f"; unverified: {', '.join(v['unverified'][:5])}" if v["unverified"] else "")))
            elif c == "citations":
                cited = sorted(set(re.findall(r"\[D(\d+)\]", answer)), key=int)
                invalid = [f"D{n}" for n in cited if not 0 < int(n) <= len(ctx["doc_sources"])]
                if invalid:
                    results.append(_result(rule, False, detail=f"cites {', '.join(invalid)} but no such passage was retrieved"))
                elif not ctx["doc_sources"]:
                    results.append(_result(rule, True, detail="no documents used"))
                else:
                    statements = [l for l in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z*])", answer)
                                  if len(re.sub(r"[*_#>\-\s]", "", l)) > 40]
                    with_cite = sum(1 for l in statements if re.search(r"\[D\d+\]", l))
                    ok = bool(cited) and with_cite >= 0.5 * len(statements)
                    detail = (f"cited {', '.join('D' + n for n in cited)}; {with_cite} of {len(statements)} statements carry a citation"
                              if cited else "documents used but not cited")
                    results.append(_result(rule, ok, detail=detail))
            elif c == "illustrative_disclosed":
                used = "rank_restoration_sites" in ctx["tools_used"]
                ok = (not used) or ("illustrative" in answer.lower())
                if not ok:
                    answer += "\n\n*Note: the restoration sites and costs used here are illustrative, not official ESRP records.*"
                results.append(_result(rule, ok, detail="illustrative site list not used" if not used
                                       else ("disclosed" if ok else "notice appended")))
            elif c == "quality_disclosed":
                issues = ctx["quality_issues"]
                if not issues:
                    results.append(_result(rule, True, detail="all data used passed quality checks"))
                else:
                    mentioned = re.search(r"quality|failed|missing|unavailable|incomplete|caveat|limitation", answer, re.I)
                    if not mentioned:
                        answer += "\n\n*Data quality note: " + "; ".join(issues[:4]) + ".*"
                    results.append(_result(rule, bool(mentioned), detail="; ".join(issues[:4])))
            elif c == "pii":
                answer, n = _redact(answer)
                results.append(_result(rule, n == 0, detail=f"{n} item(s) redacted" if n else "none found"))
        return answer, results
