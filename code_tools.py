"""Code indexing tools for SmartMemory MCP server.

Parses Python codebases locally using stdlib ast module, then POSTs
results to SmartMemory API for remote graph storage and search.

Parser adapted from smart-memory/smartmemory/code/parser.py
"""

import ast
import os
from typing import Any, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CodeEntity:
    name: str
    entity_type: str  # module | class | function | route | test
    file_path: str  # relative to repo root
    line_number: int
    repo: str
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    http_method: str = ""
    http_path: str = ""

    @property
    def item_id(self) -> str:
        return f"code::{self.repo}::{self.file_path}::{self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "docstring": self.docstring[:500] if self.docstring else "",
            "decorators": self.decorators,
            "bases": self.bases,
            "http_method": self.http_method,
            "http_path": self.http_path,
        }


@dataclass
class CodeRelation:
    source_id: str
    target_id: str
    relation_type: str  # DEFINES | IMPORTS | CALLS | INHERITS | TESTS
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
        }


@dataclass
class ParseResult:
    file_path: str
    entities: list[CodeEntity] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

ROUTER_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".tox", ".mypy_cache", ".pytest_cache"}


class CodeParser:
    def __init__(self, repo: str, repo_root: str):
        self.repo = repo
        self.repo_root = os.path.abspath(repo_root)

    def parse_file(self, file_path: str) -> ParseResult:
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, self.repo_root)
        result = ParseResult(file_path=rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            result.errors.append(f"SyntaxError in {rel_path}: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Error reading {rel_path}: {e}")
            return result
        module_name = rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        module_name = module_name.removesuffix(".__init__")
        module_entity = CodeEntity(
            name=module_name, entity_type="module", file_path=rel_path,
            line_number=1, repo=self.repo, docstring=self._get_docstring(tree),
        )
        result.entities.append(module_entity)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_class(node, module_entity, rel_path, result)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(node, module_entity, rel_path, result)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._extract_import(node, module_entity, rel_path, result)
        self._link_tests(result)
        return result

    def _extract_class(self, node, module, rel_path, result):
        bases = [self._get_name(b) for b in node.bases if self._get_name(b)]
        is_test_class = node.name.startswith("Test")
        entity = CodeEntity(
            name=node.name,
            entity_type="test" if is_test_class else "class",
            file_path=rel_path, line_number=node.lineno, repo=self.repo,
            docstring=self._get_docstring(node),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            bases=bases,
        )
        result.entities.append(entity)
        result.relations.append(CodeRelation(source_id=module.item_id, target_id=entity.item_id, relation_type="DEFINES"))
        for base_name in bases:
            base_id = self._resolve_entity_id(base_name, rel_path)
            if base_id:
                result.relations.append(CodeRelation(source_id=entity.item_id, target_id=base_id, relation_type="INHERITS"))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(child, entity, rel_path, result, class_name=node.name)

    def _extract_function(self, node, parent, rel_path, result, class_name=None):
        full_name = f"{class_name}.{node.name}" if class_name else node.name
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        route_info = self._detect_route(node)
        is_test = node.name.startswith("test_")
        if route_info:
            entity_type = "route"
        elif is_test:
            entity_type = "test"
        else:
            entity_type = "function"
        entity = CodeEntity(
            name=full_name, entity_type=entity_type, file_path=rel_path,
            line_number=node.lineno, repo=self.repo, docstring=self._get_docstring(node),
            decorators=decorators,
            http_method=route_info[0] if route_info else "",
            http_path=route_info[1] if route_info else "",
        )
        result.entities.append(entity)
        result.relations.append(CodeRelation(source_id=parent.item_id, target_id=entity.item_id, relation_type="DEFINES"))
        self._extract_calls(node, entity, rel_path, result)

    def _extract_import(self, node, module, rel_path, result):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target_module = alias.name
                target_id = f"code::{self.repo}::{self._module_to_path(target_module)}::{target_module}"
                result.relations.append(CodeRelation(
                    source_id=module.item_id, target_id=target_id,
                    relation_type="IMPORTS", properties={"names": alias.asname or alias.name},
                ))
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [a.name for a in node.names]
            target_module = node.module
            target_id = f"code::{self.repo}::{self._module_to_path(target_module)}::{target_module}"
            result.relations.append(CodeRelation(
                source_id=module.item_id, target_id=target_id,
                relation_type="IMPORTS", properties={"names": ",".join(names)},
            ))

    def _extract_calls(self, func_node, caller, rel_path, result):
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            callee_name = self._get_call_name(node)
            if not callee_name or callee_name.startswith("_"):
                continue
            target_id = f"code::{self.repo}::{rel_path}::{callee_name}"
            result.relations.append(CodeRelation(
                source_id=caller.item_id, target_id=target_id,
                relation_type="CALLS", properties={"line": getattr(node, "lineno", 0)},
            ))

    def _link_tests(self, result):
        test_entities = [e for e in result.entities if e.entity_type == "test" and "test_" in e.name]
        non_test_names = {e.name: e for e in result.entities if e.entity_type not in ("test", "module")}
        for test in test_entities:
            base_name = test.name
            if "." in base_name:
                base_name = base_name.rsplit(".", 1)[1]
            if not base_name.startswith("test_"):
                continue
            tested_name = base_name.removeprefix("test_")
            if tested_name in non_test_names:
                target = non_test_names[tested_name]
                result.relations.append(CodeRelation(
                    source_id=test.item_id, target_id=target.item_id,
                    relation_type="TESTS", properties={"convention": "name_match"},
                ))
                continue
            for fname, entity in non_test_names.items():
                if "." in fname and fname.rsplit(".", 1)[1] == tested_name:
                    result.relations.append(CodeRelation(
                        source_id=test.item_id, target_id=entity.item_id,
                        relation_type="TESTS", properties={"convention": "name_match"},
                    ))
                    break

    def _detect_route(self, node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                method = decorator.func.attr
                if method in ROUTER_METHODS and decorator.args:
                    path_arg = decorator.args[0]
                    if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                        return (method.upper(), path_arg.value)
        return None

    def _get_docstring(self, node):
        try:
            ds = ast.get_docstring(node)
            if ds:
                return ds.split("\n")[0].strip()
        except Exception:
            pass
        return ""

    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""

    def _get_decorator_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _resolve_entity_id(self, name, current_file):
        return f"code::{self.repo}::{current_file}::{name}"

    def _module_to_path(self, module_name):
        return module_name.replace(".", "/") + ".py"


def collect_python_files(directory, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    py_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return sorted(py_files)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

from server import mcp, _request, _fmt_error


@mcp.tool()
def code_index(
    directory: str,
    repo_name: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Index a Python codebase into SmartMemory's knowledge graph.

    Parses all .py files locally, extracts classes, functions, imports, routes,
    and tests, then sends to SmartMemory API for graph storage.

    Re-indexing replaces all previous entities for the same repo.

    Args:
        directory: Path to the Python project root
        repo_name: Repository identifier (default: directory basename)
        exclude_dirs: Comma-separated dirs to skip (default: __pycache__,.git,.venv,node_modules)
        workspace_id: Workspace (default: from SMARTMEMORY_WORKSPACE_ID env var)
    """
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: directory not found: {abs_dir}"

    repo = repo_name or os.path.basename(abs_dir)

    if exclude_dirs:
        excl = set(d.strip() for d in exclude_dirs.split(",") if d.strip())
    else:
        excl = DEFAULT_EXCLUDE_DIRS

    py_files = collect_python_files(abs_dir, exclude_dirs=excl)
    if not py_files:
        return f"No Python files found in {abs_dir}"

    parser = CodeParser(repo=repo, repo_root=abs_dir)
    all_entities = []
    all_relations = []
    all_errors = []

    for fpath in py_files:
        pr = parser.parse_file(fpath)
        all_entities.extend(pr.entities)
        all_relations.extend(pr.relations)
        all_errors.extend(pr.errors)

    payload = {
        "repo": repo,
        "entities": [e.to_dict() for e in all_entities],
        "relations": [r.to_dict() for r in all_relations],
    }

    result = _request("POST", "/memory/code/index", workspace_id=workspace_id, json=payload)
    err = _fmt_error(result)
    if err:
        return err
    if not isinstance(result, dict):
        return "Unexpected response from API"

    entities_stored = result.get("entities_created", len(all_entities))
    edges_stored = result.get("edges_created", len(all_relations))

    lines = [
        f"Indexed repo '{repo}' successfully.",
        f"  Files parsed: {len(py_files)}",
        f"  Entities: {entities_stored}",
        f"  Edges: {edges_stored}",
    ]
    if all_errors:
        lines.append(f"  Parse errors: {len(all_errors)}")
        for e in all_errors[:5]:
            lines.append(f"    - {e}")
        if len(all_errors) > 5:
            lines.append(f"    ... and {len(all_errors) - 5} more")
    return "\n".join(lines)


@mcp.tool()
def code_search(
    query: str,
    entity_type: Optional[str] = None,
    repo: Optional[str] = None,
    limit: int = 20,
    workspace_id: Optional[str] = None,
) -> str:
    """Search indexed code entities by name.

    Args:
        query: Search term (matches entity names, case-insensitive)
        entity_type: Filter: class, function, route, test, module (optional)
        repo: Filter by repository name (optional)
        limit: Max results (default: 20)
        workspace_id: Workspace (default: from env var)
    """
    params = {"query": query, "limit": limit}
    if entity_type:
        params["entity_type"] = entity_type
    if repo:
        params["repo"] = repo

    result = _request("GET", "/memory/code/search", workspace_id=workspace_id, params=params)
    err = _fmt_error(result)
    if err:
        return err

    # API returns a list directly
    items = result if isinstance(result, list) else []
    if not items:
        return f"No code entities found for: {query}"

    lines = [f"Found {len(items)} code entities for '{query}':\n"]
    for i, item in enumerate(items, 1):
        name = item.get("name", "?")
        etype = item.get("entity_type", "?")
        fpath = item.get("file_path", "?")
        lineno = item.get("line_number", "?")
        repo_name = item.get("repo", "")
        prefix = f"[{repo_name}] " if repo_name else ""
        line = f"{i}. {prefix}{etype}: {name}  ({fpath}:{lineno})"
        if etype == "route":
            method = item.get("http_method", "")
            path = item.get("http_path", "")
            if method and path:
                line += f"  {method} {path}"
        docstring = item.get("docstring", "")
        if docstring:
            line += f"\n   {docstring}"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def code_dependencies(
    entity_name: str,
    direction: str = "both",
    repo: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> str:
    """Trace code dependencies -- what calls/imports/inherits what.

    Args:
        entity_name: Code entity name to analyze
        direction: "dependents" (what uses this), "dependencies" (what this uses), "both"
        repo: Filter by repository (optional)
        workspace_id: Workspace (default: from env var)
    """
    params = {"entity_name": entity_name, "direction": direction}
    if repo:
        params["repo"] = repo

    result = _request("GET", "/memory/code/dependencies", workspace_id=workspace_id, params=params)
    err = _fmt_error(result)
    if err:
        return err
    if not isinstance(result, dict):
        return "Unexpected response from API"

    # API returns {"root": {...}, "dependents": [...], "dependencies": [...]}
    root = result.get("root", {})
    dependents = result.get("dependents", [])
    dependencies = result.get("dependencies", [])

    lines = []
    if root:
        etype = root.get("entity_type", "?")
        fpath = root.get("file_path", "?")
        lineno = root.get("line_number", "?")
        lines.append(f"Entity: {entity_name} ({etype}) at {fpath}:{lineno}")
    else:
        lines.append(f"Entity: {entity_name}")

    if direction in ("dependencies", "both") and dependencies:
        lines.append(f"\nDependencies ({len(dependencies)} -- what this uses):")
        for dep in dependencies:
            rel = dep.get("edge_type", "?")
            target = dep.get("name", dep.get("item_id", "?"))
            target_type = dep.get("entity_type", "")
            suffix = f" ({target_type})" if target_type else ""
            lines.append(f"  {rel} -> {target}{suffix}")
    elif direction in ("dependencies", "both"):
        lines.append("\nDependencies: none")

    if direction in ("dependents", "both") and dependents:
        lines.append(f"\nDependents ({len(dependents)} -- what uses this):")
        for dep in dependents:
            rel = dep.get("edge_type", "?")
            source = dep.get("name", dep.get("item_id", "?"))
            source_type = dep.get("entity_type", "")
            suffix = f" ({source_type})" if source_type else ""
            lines.append(f"  {source}{suffix} {rel} -> this")
    elif direction in ("dependents", "both"):
        lines.append("\nDependents: none")

    return "\n".join(lines)
