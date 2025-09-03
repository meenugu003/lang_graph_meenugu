# decision_chatbot_fixed.py
import yaml
import re
from typing import List, Literal, Optional, Dict, Any, TypedDict, Tuple
from pydantic import BaseModel, Field, root_validator, ValidationError

from langgraph.graph import StateGraph, END

# -------------------------
# Models
# -------------------------
class Option(BaseModel):
    label: str
    next: str

class ProvenanceItem(BaseModel):
    file: str
    score: float

class Provenance(BaseModel):
    __root__: Dict[str, List[ProvenanceItem]] = Field(default_factory=dict)

class Node(BaseModel):
    id: str
    kind: Literal["question", "leaf"]
    text: Optional[str] = None
    options: Optional[List[Option]] = None
    emitsPattern: Optional[str] = None

    @root_validator
    def check_consistency(cls, values):
        kind = values.get("kind")
        opts = values.get("options")
        emits = values.get("emitsPattern")
        if kind == "question" and (not opts or len(opts) == 0):
            raise ValueError("question node must have at least one option")
        if kind == "leaf" and not emits:
            raise ValueError("leaf node should have emitsPattern")
        return values

class DecisionTreeSpec(BaseModel):
    start: str
    nodes: List[Node]
    provenance: Optional[Provenance] = None

class DecisionTree(BaseModel):
    apiVersion: str
    kind: str
    metadata: Dict[str, Any]
    spec: DecisionTreeSpec

# -------------------------
# Helpers & validation
# -------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def find_matching_option(user_input: str, options: List[Option]) -> Tuple[Optional[Option], Optional[str]]:
    ui = user_input.strip()
    if ui.isdigit():
        idx = int(ui) - 1
        if 0 <= idx < len(options):
            return options[idx], "index"
        return None, None

    norm_ui = normalize(ui)
    # exact label
    for opt in options:
        if normalize(opt.label) == norm_ui:
            return opt, "label-exact"

    # substring matches
    matches = [opt for opt in options if norm_ui in normalize(opt.label)]
    if len(matches) == 1:
        return matches[0], "label-substring"
    if len(matches) > 1:
        return None, "ambiguous-substring"
    return None, None

def validate_tree_structure(tree: DecisionTree):
    nodes_by_id = {n.id: n for n in tree.spec.nodes}
    if len(nodes_by_id) != len(tree.spec.nodes):
        # find duplicates
        seen = set(); dups = set()
        for n in tree.spec.nodes:
            if n.id in seen: dups.add(n.id)
            seen.add(n.id)
        raise ValueError(f"Duplicate node ids: {sorted(list(dups))}")

    if tree.spec.start not in nodes_by_id:
        raise ValueError(f"Start node '{tree.spec.start}' not found")

    missing = []
    for n in tree.spec.nodes:
        if n.kind == "question":
            for opt in n.options:
                if opt.next not in nodes_by_id:
                    missing.append((n.id, opt.label, opt.next))
    if missing:
        msgs = ", ".join(f"{a} -> {b} -> {c}" for a,b,c in missing)
        raise ValueError("Invalid edges (point to missing nodes): " + msgs)

def detect_cycle(nodes: List[Node], start_id: str):
    graph = {n.id: [opt.next for opt in (n.options or [])] for n in nodes}
    visited = set()
    stack = []

    def dfs(u):
        if u in stack:
            idx = stack.index(u)
            return stack[idx:] + [u]
        if u in visited:
            return None
        visited.add(u); stack.append(u)
        for v in graph.get(u, []):
            cyc = dfs(v)
            if cyc:
                return cyc
        stack.pop(); return None

    return dfs(start_id)

# -------------------------
# LangGraph building
# -------------------------
class TreeState(TypedDict):
    current_node_id: str
    user_choice: Optional[str]
    output: Optional[str]
    visited: List[str]
    history: List[Dict[str, Any]]

def build_langgraph_from_tree(tree: DecisionTree):
    # Structural checks
    validate_tree_structure(tree)
    cycle = detect_cycle(tree.spec.nodes, tree.spec.start)
    if cycle:
        raise ValueError("Cycle detected: " + " -> ".join(cycle))

    nodes_by_id = {n.id: n for n in tree.spec.nodes}
    workflow = StateGraph(TreeState)  # type: ignore

    # QUESTION executor now *handles* user_choice and performs transitions
    def question_node_executor(state: TreeState):
        node_id = state["current_node_id"]
        node = nodes_by_id[node_id]

        # If no choice yet, just present the question (no state override)
        user_choice = state.get("user_choice")
        if not user_choice:
            print(f"\n[Bot] {node.text or '(no text)'}")
            for i, opt in enumerate(node.options or [], start=1):
                print(f"  {i}. {opt.label}")
            # provenance if any
            if tree.spec.provenance and node_id in (tree.spec.provenance.__root__ or {}):
                refs = tree.spec.provenance.__root__[node_id]
                print("  Supporting docs:")
                for p in refs:
                    print(f"    - {p.file} (score {p.score:.2f})")
            # No state change — let caller set user_choice and re-invoke
            return {}

        # If user_choice present, attempt to match & transition
        match_opt, reason = find_matching_option(user_choice, node.options or [])
        if match_opt:
            # add to history sequence
            history = state.get("history", [])[:]  # copy
            history.append({"node_id": node_id, "selected": match_opt.label, "matched_by": reason})
            # clear user_choice and move to next node
            return {"current_node_id": match_opt.next, "user_choice": None, "history": history}
        else:
            # ambiguous or invalid; inform user and clear user_choice so they can retry
            if reason == "ambiguous-substring":
                print("[Bot] Your input matches multiple options. Please type option number or full label.")
            else:
                print("[Bot] Invalid choice. Please type the option number or full label.")
            return {"user_choice": None}  # re-prompt on next invoke

    def leaf_node_executor(state: TreeState):
        node_id = state["current_node_id"]
        node = nodes_by_id[node_id]
        pattern = node.emitsPattern or "no-strategy"
        lines = [f"[Bot] Recommended strategic pattern: {pattern}"]
        if tree.spec.provenance and node_id in (tree.spec.provenance.__root__ or {}):
            refs = tree.spec.provenance.__root__[node_id]
            lines.append("Supporting docs:")
            for p in refs:
                lines.append(f"  - {p.file} (score {p.score:.2f})")
        output = "\n".join(lines)
        print("\n" + output)
        return {"output": output}

    # add nodes
    for node in tree.spec.nodes:
        if node.kind == "question":
            workflow.add_node(node.id, question_node_executor)
        elif node.kind == "leaf":
            workflow.add_node(node.id, leaf_node_executor)
        else:
            raise ValueError(f"Unknown node kind: {node.kind}")

    # add edges for graph validation (not used to carry state updates here)
    for node in tree.spec.nodes:
        if node.kind == "question":
            for opt in (node.options or []):
                workflow.add_edge(node.id, opt.next)
        else:
            workflow.add_edge(node.id, END)

    workflow.set_entry_point(tree.spec.start)
    compiled = workflow.compile()
    return compiled

# -------------------------
# CLI runner
# -------------------------
def run_cli(tree: DecisionTree):
    try:
        app = build_langgraph_from_tree(tree)
    except Exception as e:
        print("[ERROR] build/validate failed:", e)
        return

    state: TreeState = {
        "current_node_id": tree.spec.start,
        "user_choice": None,
        "output": None,
        "visited": [],
        "history": []
    }

    print("=== Product Decision Chatbot (CLI) ===")
    print("Type 'exit' to quit.\n")

    while True:
        # Invoke current node; executor will either display the question or process user_choice
        result = app.invoke(state, config={"recursion_limit": 50})
        if result:
            state.update(result)

        # If reached output (leaf) -> done
        if state.get("output"):
            print("\n[Session completed]")
            break

        current = state["current_node_id"]
        state["visited"].append(current)
        if len(state["visited"]) > 500:
            print("[Bot] Session appears stuck; ending.")
            break

        # If question executor printed a prompt (no user_choice processed), ask the user
        # (We assume that when user_choice was processed, current_node_id changed already.)
        user_input = input("\n[You]: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Set user_choice and loop — next invoke will process it and transition
        state["user_choice"] = user_input

    # return final state (can be persisted)
    return state

# -------------------------
# Example
# -------------------------
EXAMPLE_YAML = r"""
apiVersion: lob.ai/v1
kind: DecisionTree
metadata:
  treeId: 7a0f
  product: stream-proc
  version: 2025.09.02-1
spec:
  start: n-root
  nodes:
    - id: n-root
      kind: question
      text: "What ingestion pattern fits your workload?"
      options:
        - label: "Exactly-once streaming"
          next: n-eos
        - label: "At-least-once batch"
          next: n-batch
    - id: n-eos
      kind: question
      text: "Throughput target?"
      options:
        - label: ">= 50k events/sec"
          next: n-high
        - label: "< 50k events/sec"
          next: leaf-s1
    - id: leaf-s1
      kind: leaf
      emitsPattern: pat-EO-50k
    - id: n-batch
      kind: question
      text: "Batch size?"
      options:
        - label: "Large"
          next: leaf-s2
        - label: "Small"
          next: leaf-s3
    - id: leaf-s2
      kind: leaf
      emitsPattern: pat-AL-large
    - id: leaf-s3
      kind: leaf
      emitsPattern: pat-AL-small
    - id: n-high
      kind: question
      text: "Data source?"
      options:
        - label: "External services"
          next: leaf-s4
        - label: "Internal systems"
          next: leaf-s5
    - id: leaf-s4
      kind: leaf
      emitsPattern: pat-EO-high-ext
    - id: leaf-s5
      kind: leaf
      emitsPattern: pat-EO-high-int
"""

if __name__ == "__main__":
    try:
        data = yaml.safe_load(EXAMPLE_YAML)
        tree = DecisionTree.parse_obj(data)
    except ValidationError as e:
        print("Tree validation failed:", e)
        raise
    except Exception as e:
        print("Failed to load example:", e)
        raise

    run_cli(tree)
