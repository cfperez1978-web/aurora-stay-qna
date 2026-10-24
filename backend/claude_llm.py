# claude_llm.py
# Claude through the Claude.ai subscription (no API key).
# Every model call goes to the Claude Code CLI ("claude -p"), authenticated with the
# CLAUDE_CODE_OAUTH_TOKEN created by `claude setup-token` on the user's desktop.
import os, json, re, shutil, subprocess, tempfile, threading, time, asyncio
from langchain_core.messages import AIMessage

CLAUDE_BIN = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
_SEM = threading.Semaphore(int(os.getenv("CLAUDE_MAX_PARALLEL", "4")))   # cap parallel calls (subscription rate limits)
_WORKDIR = tempfile.mkdtemp(prefix="claude_cli_")                         # empty folder: no project files visible to the CLI
SYSTEM_PROMPT = ("You are a precise assistant. Follow the user's instructions exactly. "
                 "Do not use tools. Reply with the requested content only.")

def _tool_flags():
    """Disable all CLI tools so Claude acts as a plain text model."""
    try:
        help_txt = subprocess.run([CLAUDE_BIN, "--help"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        help_txt = ""
    if "--tools" in help_txt:
        return ["--tools", ""]
    return ["--disallowedTools", "Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch",
            "WebSearch", "Task", "NotebookEdit", "TodoWrite"]
_TOOL_FLAGS = None

def claude_cli(prompt: str, model: str = "haiku", system: str = SYSTEM_PROMPT,
               retries: int = 4, timeout: int = 300) -> str:
    """Send one prompt to Claude (model alias: haiku / sonnet / opus) and return the text reply."""
    global _TOOL_FLAGS
    if _TOOL_FLAGS is None:
        _TOOL_FLAGS = _tool_flags()
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--model", model,
           "--system-prompt", system] + _TOOL_FLAGS
    last = None
    for attempt in range(retries):
        try:
            with _SEM:
                r = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True,
                                   timeout=timeout, cwd=_WORKDIR)
            out = r.stdout.decode("utf-8", "replace").strip()
            data = json.loads(out)
            if not data.get("is_error") and data.get("result") is not None:
                return str(data["result"]).strip()
            last = data.get("result") or out
        except Exception as e:
            last = f"{type(e).__name__}: {e} | {r.stderr.decode('utf-8','replace')[:300] if 'r' in locals() else ''}"
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Claude CLI call failed after {retries} attempts: {str(last)[:500]}")


class ChatClaude:
    """Drop-in replacement for ChatOpenAI(...).invoke(prompt) -> message with .content."""
    def __init__(self, model="haiku", temperature=0.0, top_p=1.0, max_tokens=512, **kwargs):
        self.model, self.temperature, self.top_p, self.max_tokens = model, temperature, top_p, max_tokens
    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        # The CLI does not expose sampling or length controls, so the answer length budget is stated in the prompt.
        if self.max_tokens:
            text += f"\n\n(Keep the reply under about {int(self.max_tokens * 0.75)} words.)"
        return AIMessage(content=claude_cli(text, self.model))


def _extract_json(txt: str) -> str:
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    s, e = txt.find("{"), txt.rfind("}")
    return txt[s:e + 1] if s != -1 and e != -1 else txt

try:
    from deepeval.models import DeepEvalBaseLLM

    class ClaudeJudge(DeepEvalBaseLLM):
        """DeepEval judge model backed by Claude (default Sonnet) through the subscription CLI."""
        def __init__(self, model="sonnet"):
            self.model_name = model
            super().__init__(model)
        def load_model(self):
            return self
        def generate(self, prompt, schema=None, **kwargs):
            if schema is None:
                return claude_cli(prompt, self.model_name)
            ask = (prompt + "\n\nReturn ONLY one valid JSON object that matches this JSON schema. "
                   "No prose, no code fences.\n" + json.dumps(schema.model_json_schema()))
            last = None
            for _ in range(3):
                txt = claude_cli(ask, self.model_name)
                try:
                    return schema.model_validate_json(_extract_json(txt))
                except Exception as e:
                    last = e
            raise last
        async def a_generate(self, prompt, schema=None, **kwargs):
            return await asyncio.to_thread(self.generate, prompt, schema)
        def get_model_name(self):
            return f"Claude {self.model_name} (Claude.ai subscription via Claude Code CLI)"
except ImportError:      # the deployed backend does not need DeepEval
    pass
