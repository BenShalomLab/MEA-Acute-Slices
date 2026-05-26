// Expression-based filter engine.
//
// Grammar (recursive descent, case-insensitive keywords):
//
//   expr      := or_expr
//   or_expr   := and_expr  ( ( "OR"  | "||" )  and_expr )*
//   and_expr  := unary     ( ( "AND" | "&&"  ) unary    )*
//   unary     := ( "NOT" | "!" ) unary
//              | atom
//   atom      := "(" expr ")"
//              | rule
//   rule      := IDENT op operand
//   op        := "==" | "=" | "!=" | "<>" | "<" | "<=" | ">" | ">="
//              | "contains" | "!contains" | "not contains"
//              | "starts with" | "ends with"
//              | "in" | "!in" | "not in"
//              | "between"
//              | "is" | "is not"
//              | "before" | "after" | "on or before" | "on or after"
//   operand   := list | between_operand | value
//   list      := "[" value ("," value)* "]"
//   between_operand := value ("AND"|"and"|",") value
//   value     := STRING | NUMBER | "true" | "false" | IDENT
//
// Compiles to AST node shapes used by the visual readout and evaluator:
//   { kind: "rule",  not, field, op, value }     -- value: string | number | array | [a, b] for between
//   { kind: "and"|"or", children: [...] }
//   { kind: "not",   child }

(function () {
  const F = {};

  // ── Tokenizer ────────────────────────────────────────────────────
  // Tokens: PAREN("(" or ")"), BRACKET("[" or "]"), COMMA, AND, OR, NOT,
  //         OP (multi-char operators), KW (in/contains/between/is/before/after/and-in-between),
  //         IDENT, STRING, NUMBER, BOOL
  const KEYWORDS = {
    "and":  "AND",
    "&&":   "AND",
    "or":   "OR",
    "||":   "OR",
    "not":  "NOT",
    "!":    "NOT",                    // standalone ! prefix (NOT)
    "in":   "OP", "!in": "OP", "not in": "OP",
    "contains": "OP", "!contains": "OP", "not contains": "OP",
    "starts with": "OP", "ends with": "OP",
    "between": "OP",
    "is": "OP", "is not": "OP",
    "before": "OP", "after": "OP", "on or before": "OP", "on or after": "OP",
    "true":  "BOOL", "false": "BOOL", "null": "BOOL",
  };

  // Order matters: longer multi-word ops first
  const MULTI_WORD = [
    "on or before", "on or after",
    "not contains", "not in",
    "is not",
    "starts with", "ends with",
  ];

  function tokenize(src) {
    const toks = [];
    let i = 0, line = 1, col = 1;
    const N = src.length;

    function push(type, value, len) {
      toks.push({ type, value, pos: i, line, col, len });
      for (let k = 0; k < len; k++) {
        if (src[i + k] === "\n") { line++; col = 1; } else col++;
      }
      i += len;
    }

    while (i < N) {
      const ch = src[i];
      // whitespace
      if (/\s/.test(ch)) {
        if (ch === "\n") { line++; col = 1; } else col++;
        i++; continue;
      }
      // parens / brackets / comma
      if (ch === "(" || ch === ")") { push("PAREN", ch, 1); continue; }
      if (ch === "[" || ch === "]") { push("BRACKET", ch, 1); continue; }
      if (ch === ",")               { push("COMMA", ",", 1); continue; }

      // strings
      if (ch === '"' || ch === "'") {
        let j = i + 1;
        while (j < N && src[j] !== ch) {
          if (src[j] === "\\" && j + 1 < N) j += 2; else j++;
        }
        if (j >= N) throw makeErr(`Unterminated string starting at column ${col}`, i, j - i);
        const raw = src.slice(i + 1, j).replace(/\\(.)/g, "$1");
        push("STRING", raw, j - i + 1);
        continue;
      }

      // ISO date literal (treated as STRING so date comparisons work)
      const dateMatch = /^\d{4}-\d{2}-\d{2}/.exec(src.slice(i));
      if (dateMatch) { push("STRING", dateMatch[0], dateMatch[0].length); continue; }

      // numbers (including negatives if at start of an operand position — we
      // handle that here by allowing leading - if previous token wasn't an
      // operand/closing paren).
      const numMatch = /^-?\d+(\.\d+)?/.exec(src.slice(i));
      if (numMatch) {
        const prev = toks[toks.length - 1];
        const canBeNegative = !prev
          || prev.type === "AND" || prev.type === "OR" || prev.type === "NOT"
          || (prev.type === "PAREN" && prev.value === "(")
          || (prev.type === "BRACKET" && prev.value === "[")
          || prev.type === "COMMA"
          || prev.type === "OP";
        if (numMatch[0].startsWith("-") && !canBeNegative) {
          // treat as bare "-" — not supported; fall through to error below
        } else {
          push("NUMBER", parseFloat(numMatch[0]), numMatch[0].length);
          continue;
        }
      }

      // multi-char comparison operators
      const ops2 = ["==", "!=", "<=", ">=", "<>"];
      let matched = false;
      for (const op of ops2) {
        if (src.startsWith(op, i)) { push("OP", op, op.length); matched = true; break; }
      }
      if (matched) continue;
      if (ch === "=" || ch === "<" || ch === ">") { push("OP", ch === "=" ? "==" : ch, 1); continue; }

      // identifier OR keyword
      const idMatch = /^[A-Za-z_][A-Za-z0-9_.]*/.exec(src.slice(i));
      if (idMatch) {
        const raw = idMatch[0];
        const lower = raw.toLowerCase();

        // Try multi-word keyword starting here
        let mw = null;
        for (const k of MULTI_WORD) {
          const re = new RegExp("^" + k.replace(/\s+/g, "\\s+"), "i");
          const m = re.exec(src.slice(i));
          if (m) { mw = { word: k, len: m[0].length }; break; }
        }
        if (mw) { push("OP", mw.word, mw.len); continue; }

        if (lower === "and") { push("AND", "AND", raw.length); continue; }
        if (lower === "or")  { push("OR",  "OR",  raw.length); continue; }
        if (lower === "not") { push("NOT", "NOT", raw.length); continue; }
        if (KEYWORDS[lower] === "OP")   { push("OP", lower, raw.length); continue; }
        if (KEYWORDS[lower] === "BOOL") { push("BOOL", lower === "true", raw.length); continue; }

        push("IDENT", raw, raw.length);
        continue;
      }

      // !  -- could be NOT or part of !contains/!in; ambiguous, handle as NOT here
      if (ch === "!") {
        // check for !contains / !in directly (handled above by MULTI_WORD? no — those were "not in"/"not contains")
        const ahead = src.slice(i + 1, i + 12).toLowerCase();
        if (ahead.startsWith("contains")) { push("OP", "!contains", "!contains".length); continue; }
        if (ahead.startsWith("in") && !/^in\w/.test(ahead)) { push("OP", "!in", "!in".length); continue; }
        push("NOT", "NOT", 1);
        continue;
      }

      // unknown
      throw makeErr(`Unexpected character "${ch}" at column ${col}`, i, 1);
    }
    return toks;

    function makeErr(msg, pos, len) {
      const e = new Error(msg);
      e.pos = pos; e.len = len; e.line = line; e.col = col;
      return e;
    }
  }

  // ── Parser ───────────────────────────────────────────────────────
  function parse(src, schema) {
    const toks = tokenize(src);
    let p = 0;

    function peek(off = 0) { return toks[p + off]; }
    function eat(type, value) {
      const t = toks[p];
      if (!t) throw err(`Expected ${type}${value ? ` "${value}"` : ""} but got end of input`, toks[p - 1]);
      if (t.type !== type || (value != null && (t.value + "").toLowerCase() !== (value + "").toLowerCase())) {
        throw err(`Expected ${type}${value ? ` "${value}"` : ""} but got "${t.value}"`, t);
      }
      p++; return t;
    }
    function err(msg, t) {
      const e = new Error(msg);
      e.pos = t ? t.pos : 0;
      e.len = t ? t.len : 1;
      return e;
    }

    function parseExpr() { return parseOr(); }
    function parseOr() {
      let left = parseAnd();
      const parts = [left];
      while (peek() && peek().type === "OR") { p++; parts.push(parseAnd()); }
      return parts.length === 1 ? left : { kind: "or", children: parts };
    }
    function parseAnd() {
      let left = parseUnary();
      const parts = [left];
      while (peek() && peek().type === "AND") { p++; parts.push(parseUnary()); }
      return parts.length === 1 ? left : { kind: "and", children: parts };
    }
    function parseUnary() {
      if (peek() && peek().type === "NOT") { p++; return { kind: "not", child: parseUnary() }; }
      return parseAtom();
    }
    function parseAtom() {
      const t = peek();
      if (!t) throw err("Unexpected end of expression — did you forget the right-hand value?", toks[p - 1]);
      if (t.type === "PAREN" && t.value === "(") {
        p++;
        const inner = parseExpr();
        const close = peek();
        if (!close || close.type !== "PAREN" || close.value !== ")") {
          throw err("Missing closing parenthesis ')'", close || t);
        }
        p++;
        return inner;
      }
      return parseRule();
    }
    function parseRule() {
      const fT = peek();
      if (!fT || fT.type !== "IDENT") {
        throw err(`Expected a field name (e.g. ${schema.slice(0,3).map(s=>s.field).join(", ")}…) but got "${fT?.value ?? "?"}"`, fT);
      }
      p++;
      // Validate field
      const meta = schema.find((s) => s.field === fT.value.toLowerCase() || s.field === fT.value);
      if (!meta) {
        const known = schema.map((s) => s.field).slice(0, 6).join(", ");
        throw err(`Unknown field "${fT.value}". Known fields: ${known}…`, fT);
      }

      const opT = peek();
      if (!opT || opT.type !== "OP") {
        throw err(`Expected an operator after field "${meta.field}" (e.g. ==, contains, in)`, opT || fT);
      }
      p++;

      let opVal = opT.value;
      // Normalize aliases
      const opMap = {
        "=": "==", "<>": "!=",
        "not contains": "!contains",
        "not in": "!in",
        "before": "<", "after": ">",
        "on or before": "<=", "on or after": ">=",
        "is not": "is",         // operand will be the inverted value
      };
      const wasIsNot = opT.value.toLowerCase() === "is not";
      if (opMap[opVal.toLowerCase()]) opVal = opMap[opVal.toLowerCase()];

      // operand parsing
      let value;
      if (opVal === "between") {
        const a = parseValue();
        const tAnd = peek();
        if (!tAnd || (tAnd.type !== "AND" && !(tAnd.type === "COMMA"))) {
          throw err(`"between" needs two values joined by AND, e.g. ${meta.field} between 10 and 20`, tAnd || opT);
        }
        p++;
        const b = parseValue();
        value = [a, b];
      } else if (opVal === "in" || opVal === "!in") {
        const lb = peek();
        if (lb && lb.type === "BRACKET" && lb.value === "[") {
          p++;
          const arr = [];
          while (peek() && !(peek().type === "BRACKET" && peek().value === "]")) {
            arr.push(parseValue());
            if (peek() && peek().type === "COMMA") p++;
          }
          if (!peek() || peek().type !== "BRACKET" || peek().value !== "]") {
            throw err("Missing closing bracket ']'", peek() || opT);
          }
          p++;
          value = arr;
        } else {
          // bare list: in 1, 2, 3
          const arr = [parseValue()];
          while (peek() && peek().type === "COMMA") { p++; arr.push(parseValue()); }
          value = arr;
        }
      } else if (opVal === "is") {
        const v = parseValue();
        value = wasIsNot ? !v : v;
        opVal = "is";   // canonical
      } else {
        value = parseValue();
      }

      return { kind: "rule", not: false, field: meta.field, op: opVal, value };
    }
    function parseValue() {
      const t = peek();
      if (!t) throw err("Expected a value", toks[p - 1]);
      if (t.type === "STRING") { p++; return t.value; }
      if (t.type === "NUMBER") { p++; return t.value; }
      if (t.type === "BOOL")   { p++; return t.value; }
      if (t.type === "IDENT")  { p++; return t.value; } // unquoted value
      throw err(`Expected a value here, got "${t.value}"`, t);
    }

    if (toks.length === 0) return null;
    const ast = parseExpr();
    if (p < toks.length) {
      throw err(`Unexpected token "${toks[p].value}" after expression`, toks[p]);
    }
    return ast;
  }

  // ── Evaluator ────────────────────────────────────────────────────
  function coerce(v, type) {
    if (v == null) return v;
    if (type === "number" || type === "int") {
      const n = Number(v);
      return Number.isFinite(n) ? n : NaN;
    }
    if (type === "bool") return v === true || v === "true" || v === 1;
    return String(v);
  }
  function cmp(a, b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }
  function evalRule(rule, row, schema) {
    const meta = schema.find((s) => s.field === rule.field);
    if (!meta) return false;
    const left = row[rule.field];
    const type = meta.type;
    const v    = rule.value;
    switch (rule.op) {
      case "==":  return cmp(coerce(left, type), coerce(v, type)) === 0;
      case "!=":  return cmp(coerce(left, type), coerce(v, type)) !== 0;
      case "<":   return cmp(coerce(left, type), coerce(v, type)) <  0;
      case "<=":  return cmp(coerce(left, type), coerce(v, type)) <= 0;
      case ">":   return cmp(coerce(left, type), coerce(v, type)) >  0;
      case ">=":  return cmp(coerce(left, type), coerce(v, type)) >= 0;
      case "contains":  return String(left ?? "").toLowerCase().includes(String(v ?? "").toLowerCase());
      case "!contains": return !String(left ?? "").toLowerCase().includes(String(v ?? "").toLowerCase());
      case "starts with": return String(left ?? "").toLowerCase().startsWith(String(v ?? "").toLowerCase());
      case "ends with":   return String(left ?? "").toLowerCase().endsWith(String(v ?? "").toLowerCase());
      case "in":  return (v || []).some((x) => cmp(coerce(left, type), coerce(x, type)) === 0);
      case "!in": return !(v || []).some((x) => cmp(coerce(left, type), coerce(x, type)) === 0);
      case "between": {
        const [a, b] = v || [];
        const x = coerce(left, type);
        return cmp(x, coerce(a, type)) >= 0 && cmp(x, coerce(b, type)) <= 0;
      }
      case "is": return !!left === !!v;
      default: return true;
    }
  }
  function evalAst(node, row, schema) {
    if (!node) return true;
    if (node.kind === "rule") return evalRule(node, row, schema);
    if (node.kind === "not")  return !evalAst(node.child, row, schema);
    if (node.kind === "and")  return node.children.every((c) => evalAst(c, row, schema));
    if (node.kind === "or")   return node.children.some((c) => evalAst(c, row, schema));
    return true;
  }
  F.apply = function (rows, src, schema) {
    if (!src || !src.trim()) return rows;
    let ast;
    try { ast = parse(src, schema); } catch (e) { return rows; } // invalid → no filter
    if (!ast) return rows;
    return rows.filter((row) => evalAst(ast, row, schema));
  };

  // ── Pretty-printer (for the synced AST readout) ──────────────────
  function prettyValue(v, type) {
    if (Array.isArray(v)) return "[" + v.map((x) => prettyValue(x, type)).join(", ") + "]";
    if (typeof v === "string") return `"${v}"`;
    return String(v);
  }
  function tokenizePretty(node, schema, top = true) {
    const out = [];
    if (!node) return out;
    if (node.kind === "rule") {
      const meta = schema.find((s) => s.field === node.field);
      out.push({ t: "fld", s: meta?.label ?? node.field });
      out.push({ t: "text", s: " " });
      out.push({ t: "op", s: opLabel(node.op, meta?.type) });
      out.push({ t: "text", s: " " });
      if (node.op === "between") {
        out.push({ t: "val", s: prettyValue(node.value?.[0], meta?.type) });
        out.push({ t: "text", s: " " });
        out.push({ t: "and", s: "AND" });
        out.push({ t: "text", s: " " });
        out.push({ t: "val", s: prettyValue(node.value?.[1], meta?.type) });
      } else {
        out.push({ t: "val", s: prettyValue(node.value, meta?.type) });
      }
      return out;
    }
    if (node.kind === "not") {
      out.push({ t: "not", s: "NOT " });
      const inner = tokenizePretty(node.child, schema, false);
      if (node.child && node.child.kind !== "rule") {
        out.push({ t: "paren", s: "(" });
        out.push(...inner);
        out.push({ t: "paren", s: ")" });
      } else {
        out.push(...inner);
      }
      return out;
    }
    // and / or
    const wrap = !top;
    if (wrap) out.push({ t: "paren", s: "(" });
    node.children.forEach((c, i) => {
      if (i > 0) {
        out.push({ t: "text", s: " " });
        out.push({ t: node.kind, s: node.kind.toUpperCase() });
        out.push({ t: "text", s: " " });
      }
      out.push(...tokenizePretty(c, schema, false));
    });
    if (wrap) out.push({ t: "paren", s: ")" });
    return out;
  }
  function opLabel(op, type) {
    if (type === "date") {
      const m = { "<": "earlier than", "<=": "on or earlier than", ">": "later than", ">=": "on or later than", "==": "is", "!=": "is not" };
      return m[op] || op;
    }
    if (type === "number" || type === "int") {
      const m = { "<": "less than", "<=": "≤", ">": "greater than", ">=": "≥", "==": "equals", "!=": "≠" };
      return m[op] || op;
    }
    const m = { "==": "exactly", "!=": "not exactly" };
    return m[op] || op;
  }

  // ── Public surface ───────────────────────────────────────────────
  F.parse = parse;
  F.tokenize = tokenize;
  F.tokenizePretty = tokenizePretty;
  F.evalAst = evalAst;
  F.opLabel = opLabel;
  window.FilterEng = F;
})();
