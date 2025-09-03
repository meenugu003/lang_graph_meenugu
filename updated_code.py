"""
decision_chatbot.py

Production-hardened Decision Tree -> LangGraph builder + CLI runner.

Assumptions:
- langgraph.graph.StateGraph API supports add_node, add_edge, add_conditional_edges,
  set_entry_point, compile, and compiled app.invoke(state, config=...).
- pydantic is available for validation.
- yaml is available to load YAML examples.

This module:
- Defines Pydantic models for trees
- Validates tree structure (unique ids, question nodes have options, edges point to nodes)
- Detects cycles at compile-time
- Builds a LangGraph StateGraph with safe executors
- Provides a friendly router with robust input matching
- Demonstrates a simple CLI runner (replaceable by API handlers)
"""

import yaml
import re
from typing import List, Literal, Optional, Dict, Any, TypedDict, Set, Tuple
from pydantic import BaseModel, Field, ValidationError, root_validator
from collections import deque

# Replace or adapt these if your langgraph API differs slightly
from langgraph.graph import StateGraph, END
# from langgraph.graph import START  # If START constant is needed; not used here
# from langchain_core.messages import HumanMessage  # not required for this demo

# -------------------------
# Pydantic data models
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
    def check_leaf_and_question_consistency(cls, values):
        kind = values.get("kind")
        options = values.get("options")
        emits = values.get("emitsPattern")
        if kind == "question" and (not options or len(options) == 0):
            raise ValueError("question node must have at least one option")
        if kind == "leaf" and not emits:
            raise ValueError("leaf node should have emitsPattern (strategy identifier)")
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
# Utility validators
# -------------------------
def validate_tree_structure(tree: DecisionTree) -> None:
    """
    Validates:
     - Unique node IDs
     - Options reference existing nodes
     - Start node exists
    Raises ValueError on issues.
    """
    nodes_by_id = {n.id: n for n in tree.spec.nodes}
    if len(nodes_by_id) != len(tree.spec.nodes):
        # find duplicates
        seen = set()
        dups = set()
        for n in tree.spec.nodes:
            if n.id in seen:
                dups.add(n.id)
            seen.add(n.id)
        raise ValueError(f"Duplicate node ids found: {sorted(list(dups))}")

    if tree.spec.start not in nodes_by_id:
        raise ValueError(f"Start node '{tree.spec.start}' not found in nodes")

    # Validate options point to real nodes
    missing_refs = []
    for node in tree.spec.nodes:
        if node.kind == "question":
            for opt in node.options:
                if opt.next not in nodes_by_id:
                    missing_refs.append((node.id, opt.label, opt.next))

    if missing_refs:
        msgs = ", ".join([f"{src} -> {lbl} -> {dst}" for src, lbl, dst in missing_refs])
        raise ValueError(f"Invalid edges found (point to missing node): {msgs}")

def detect_cycle(nodes: List[Node], start_id: str) -> Optional[List[str]]:
    """
    Detects cycles reachable from start using DFS.
    Returns one cycle path list if found, else None.
    """
    graph = {n.id: [opt.next for opt in (n.options or [])] for n in nodes}
    visited = set()
    stack = []

    def dfs(u):
        if u in stack:
            # cycle is from first occurrence of u in stack to end
            idx = stack.index(u)
            return stack[idx:] + [u]
        if u in visited:
            return None
        visited.add(u)
        stack.append(u)
        for v in graph.get(u, []):
            cyc = dfs(v)
            if cyc:
                return cyc
        stack.pop()
        return None

    return dfs(start_id)

# -------------------------
# Router / Matching utils
# -------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def find_matching_option(user_input: str, options: List[Option]) -> Tuple[Optional[Option], Optional[str]]:
    """
    Tries to match user_input to an Option.
    Returns (option, reason) where reason explains the matching method or None if ambiguous.
    Matching strategy:
     - Exact index match (1,2,...)
     - Exact label (case-insensitive)
     - Label substring match (if unique)
    """
    ui = user_input.strip()
    if ui.isdigit():
        idx = int(ui) - 1
        if 0 <= idx < len(options):
            return options[idx], "index"
        return None, None

    norm_ui = normalize(ui)
    # exact label match
    for opt in options:
        if normalize(opt.label) == norm_ui:
            return opt, "label-exact"

    # substring matches (case-insensitive)
    matches = []
    for opt in options:
        if norm_ui in normalize(opt.label):
            matches.append(opt)

    if len(matches) == 1:
        return matches[0], "label-substring"
    if len(matches) > 1:
        return None, "ambiguous-substring"

    return None, None

# -------------------------
# LangGraph builder
# -------------------------
class TreeState(TypedDict):
    current_node_id: str
    user_choice: Optional[str]
    output: Optional[str]
    visited: List[str]  # track visited node ids (sequence)
    history: List[Dict[str, Any]]  # list of {node_id, selected_option_label, matched_by}

def build_langgraph_from_tree(tree: DecisionTree) -> Any:
    """
    Validates and compiles a DecisionTree into a LangGraph StateGraph.
    Returns the compiled graph object.
    """
    # Basic structural validation
    validate_tree_structure(tree)

    # Cycle detection (at least report)
    cycle = detect_cycle(tree.spec.nodes, tree.spec.start)
    if cycle:
        # don't fail hard if you want cycles intentionally, but warn
        raise ValueError(f"Cycle detected in decision tree: {' -> '.join(cycle)}")

    nodes_by_id = {node.id: node for node in tree.spec.nodes}

    # Build a StateGraph. We assume StateGraph accepts a TypedDict type hint for state
    workflow = StateGraph(TreeState)  # type: ignore

    # Executor for question nodes
    def question_node_executor(state: TreeState):
        node_id = state["current_node_id"]
        node = nodes_by_id[node_id]
        # Print question and options for CLI; in API, replace this with returned payload
        display_text = node.text or "(no question text)"
        print(f"\n[Bot] {display_text}")
        for i, opt in enumerate(node.options or [], start=1):
            print(f"  {i}. {opt.label}")
        # Provide provenance hint if exists
        prov_text = ""
        if tree.spec.provenance and node_id in (tree.spec.provenance.__root__ or {}):
            refs = tree.spec.provenance.__root__[node_id]
            prov_text = "\n  Supporting docs:\n" + "\n".join(f"    - {p.file} (score {p.score:.2f})" for p in refs)
            print(prov_text)
        # Return state as-is; router will use state['user_choice'] to decide transition
        return {"current_node_id": node_id}

    # Executor for leaf nodes
    def leaf_node_executor(state: TreeState):
        node_id = state["current_node_id"]
        node = nodes_by_id[node_id]
        pattern = node.emitsPattern or "no-strategy"
        output_lines = [f"[Bot] Recommended strategic pattern: {pattern}"]

        # Attach provenance if available for this leaf
        if tree.spec.provenance and node_id in (tree.spec.provenance.__root__ or {}):
            refs = tree.spec.provenance.__root__[node_id]
            output_lines.append("Supporting docs:")
            for p in refs:
                output_lines.append(f"  - {p.file} (score {p.score:.2f})")

        output_text = "\n".join(output_lines)
        print("\n" + output_text)
        # set output in state so CLI/main loop can break
        return {"output": output_text}

    # Router that decides next node id
    def router(state: TreeState):
        current_node_id = state["current_node_id"]
        node = nodes_by_id[current_node_id]

        user_choice = state.get("user_choice")
        # If no user choice supplied, stay on current node so question executor can re-run / prompt
        if not user_choice:
            return current_node_id

        # Prevent accidental infinite repeats: if visited too many times for the same node, bail
        visited = state.get("visited", [])
        if visited.count(current_node_id) > 5:
            # too many re-visits; failover to end or raise
            print(f"[Bot] Too many attempts at node {current_node_id}. Aborting.")
            return current_node_id

        # Try to match the option
        match_opt, reason = find_matching_option(user_choice, node.options or [])
        if match_opt is not None:
            # update visited/history are handled externally; just return next node id
            return match_opt.next

        # Ambiguous substring -> inform user to be more specific
        if reason == "ambiguous-substring":
            print("[Bot] Your input matches multiple options. Please type the option number or full label.")
            return current_node_id

        # No match -> inform user and re-prompt
        print("[Bot] Invalid choice. Please reply with the option number or a clear label.")
        return current_node_id

    # Add nodes to workflow
    for node in tree.spec.nodes:
        if node.kind == "question":
            workflow.add_node(node.id, question_node_executor)
        elif node.kind == "leaf":
            workflow.add_node(node.id, leaf_node_executor)
        else:
            raise ValueError(f"Unknown node kind for node {node.id}: {node.kind}")

    # Add edges for conditionals for question nodes
    # We map each option.next to that node id; router will return the node id string to select edge.
    for node in tree.spec.nodes:
        if node.kind == "question":
            # Create mapping opt.next -> opt.next (langgraph expects dict of allowed nexts -> labels/ids)
            next_map = {opt.next: opt.next for opt in (node.options or [])}
            workflow.add_conditional_edges(node.id, router, next_map)
        elif node.kind == "leaf":
            # leaf -> END
            workflow.add_edge(node.id, END)

    # Set start point
    workflow.set_entry_point(tree.spec.start)
    # Compile (returns compiled object that has .invoke)
    compiled = workflow.compile()
    return compiled

# -------------------------
# CLI runner (demo)
# -------------------------
def run_cli(tree: DecisionTree):
    """
    Simple CLI runner using compiled LangGraph. Replace with API handlers as needed.
    """
    try:
        app = build_langgraph_from_tree(tree)
    except Exception as e:
        print(f"[ERROR] Tree validation/compile failed: {e}")
        return

    # state initialization
    state: TreeState = {
        "current_node_id": tree.spec.start,
        "user_choice": None,
        "output": None,
        "visited": [],
        "history": []
    }

    print("=== Unified Product Decision Chatbot (CLI demo) ===")
    print("Type 'exit' to quit. Type the option number or label to choose.\n")

    while True:
        # invoke will execute the node function (question or leaf)
        result = app.invoke(state, config={"recursion_limit": 50})
        # Merge result into state (LangGraph nodes return partial updates)
        if result:
            for k, v in result.items():
                state[k] = v

        # If output produced -> finished
        if state.get("output"):
            print("\n[Session complete]")
            break

        current_node_id = state["current_node_id"]
        # Run-time protection: detect if we've visited the same node too many times in session
        state["visited"].append(current_node_id)
        if len(state["visited"]) > 200:
            print("[Bot] Session appears stuck (too many steps). Ending session.")
            break

        # Wait for user input (CLI). In API mode, this would be provided by client per request.
        user_input = input("\n[You]: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Save to history for resume capability (could be persisted externally)
        state["history"].append({
            "node_id": current_node_id,
            "raw_input": user_input
        })

        # Set user_choice and loop (router will use it)
        state["user_choice"] = user_input

    # End of session. Optionally return state for persistence.
    return state

# -------------------------
# Example YAML and main guard
# -------------------------
EXAMPLE_YAML = r"""
apiVersion: lob.ai/v1
kind: DecisionTree
metadata:
  treeId: 7a0f
  product: stream-proc
  version: 2025.09.02-1
  sourceBundle: bndl-123
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
  provenance:
    n-eos:
      - file: md/ingestion.md#exactly-once
        score: 0.83
    leaf-s4:
      - file: md/ingestion.md#external
        score: 0.74
"""

if __name__ == "__main__":
    try:
        data = yaml.safe_load(EXAMPLE_YAML)
        tree = DecisionTree.parse_obj(data)
    except ValidationError as e:
        print("YAML -> model validation failed:", e)
        raise
    except Exception as e:
        print("Failed to load example YAML:", e)
        raise

    session_state = run_cli(tree)
    # Optionally persist session_state (current_node_id, history) to Postgres/LangMem for resume.
