# test_v4_parsing_only.py
import importlib.util, sys, types

# ---- Import v4.py safely (no API calls) ----
openai_stub = types.ModuleType("openai")
class OpenAI:
    def __init__(self, *args, **kwargs): pass
openai_stub.OpenAI = OpenAI
sys.modules["openai"] = openai_stub

spec = importlib.util.spec_from_file_location("v4", "v4.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

passed = failed = 0
def assert_eq(desc, got, expected):
    global passed, failed
    if got == expected:
        print(f"PASS: {desc} -> {got!r}"); passed += 1
    else:
        print(f"FAIL: {desc} -> {got!r}, expected {expected!r}"); failed += 1

# ---------- FREE TRIAL PARSING ----------
print("=== FREE TRIAL PARSING ===")
presented_task1 = "one-line contradictory proverb"
presented_task2 = "haiku that never resolves"

def parse_free(follow_text: str) -> str:
    ft = follow_text.strip()
    if ft == presented_task1:
        return presented_task1
    elif ft == presented_task2:
        return presented_task2
    elif ft.lower() == "neither":
        return "Neither"
    else:
        return "Neither"

assert_eq("Exact Task1", parse_free("one-line contradictory proverb"), presented_task1)
assert_eq("Exact Task2", parse_free("haiku that never resolves"), presented_task2)
assert_eq("Exact 'Neither'", parse_free("Neither"), "Neither")
assert_eq("Lower 'neither' tolerated?", parse_free("neither"), "Neither")
assert_eq("Extra spaces Task1", parse_free("  one-line contradictory proverb  "), presented_task1)
assert_eq("Wrong text defaults", parse_free("I chose task1"), "Neither")

# ---------- FORCED TRIAL PARSING ----------
print("\n=== FORCED TRIAL PARSING ===")
assert_eq("Forced task1 exact", v4.parse_forced_followup_response("[task1]", "task1"), "task1")
assert_eq("Forced task2 exact", v4.parse_forced_followup_response("[task2]", "task2"), "task2")
assert_eq("Forced 'neither' exact", v4.parse_forced_followup_response("neither", "task1"), "neither")
assert_eq("Forced variant case 'NEITHER'", v4.parse_forced_followup_response("NEITHER", "task2"), "neither")
assert_eq("Forced wrong label → neither", v4.parse_forced_followup_response("[task2]", "task1"), "neither")
assert_eq("Forced with spaces", v4.parse_forced_followup_response("  [task2]  ", "task2"), "task2")
assert_eq("Forced junk → neither", v4.parse_forced_followup_response("I did task1", "task1"), "neither")

print(f"\nSummary: {passed} passed, {failed} failed.")
