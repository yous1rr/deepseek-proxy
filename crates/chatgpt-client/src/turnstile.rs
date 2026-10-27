//! Restricted interpreter for the numeric Turnstile bytecode in `utils/turnstile.py`.
//! It interprets JSON instructions only; no JavaScript or dynamic code is evaluated.
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::{Number, Value};
use std::{collections::HashMap, time::Instant};

const MAX_PROGRAM_BYTES: usize = 256 * 1024;
const MAX_INSTRUCTIONS: usize = 20_000;
const MAX_TEXT: usize = 256 * 1024;
const MAX_RECURSION: usize = 32;

#[derive(Clone, Debug, PartialEq)]
enum Vm {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Vm>),
    Object(Vec<(String, Vm)>),
    OrderedMap(Vec<(String, Vm)>),
    Function(u64),
}

fn from_json(value: &Value) -> Vm {
    match value {
        Value::Null => Vm::Null,
        Value::Bool(v) => Vm::Bool(*v),
        Value::Number(v) => Vm::Number(v.clone()),
        Value::String(v) => Vm::String(v.clone()),
        Value::Array(v) => Vm::Array(v.iter().map(from_json).collect()),
        Value::Object(v) => Vm::Object(v.iter().map(|(k, x)| (k.clone(), from_json(x))).collect()),
    }
}

fn special_string(value: &str) -> &str {
    match value {
        "window.Math" => "[object Math]",
        "window.Reflect" => "[object Reflect]",
        "window.performance" => "[object Performance]",
        "window.localStorage" => "[object Storage]",
        "window.Object" => "function Object() { [native code] }",
        "window.Reflect.set" => "function set() { [native code] }",
        "window.performance.now" => "function () { [native code] }",
        "window.Object.create" => "function create() { [native code] }",
        "window.Object.keys" => "function keys() { [native code] }",
        "window.Math.random" => "function random() { [native code] }",
        other => other,
    }
}

fn py_string(value: &Vm) -> String {
    match value {
        Vm::Null => "undefined".into(),
        Vm::Bool(v) => if *v { "True" } else { "False" }.into(),
        Vm::Number(v) => v.to_string(),
        Vm::String(v) => special_string(v).into(),
        Vm::Array(values) if values.iter().all(|x| matches!(x, Vm::String(_))) => {
            values.iter().map(py_string).collect::<Vec<_>>().join(",")
        }
        Vm::Array(values) => format!(
            "[{}]",
            values.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Vm::Object(_) | Vm::OrderedMap(_) => "[object Object]".into(),
        Vm::Function(_) => "[native code]".into(),
    }
}
fn py_repr(value: &Vm) -> String {
    match value {
        Vm::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        _ => py_string(value),
    }
}

fn xor_string(text: &str, key: &str) -> Option<String> {
    if key.is_empty() {
        return Some(text.to_owned());
    }
    let keys: Vec<u32> = key.chars().map(u32::from).collect();
    text.chars()
        .enumerate()
        .map(|(i, ch)| char::from_u32(u32::from(ch) ^ keys[i % keys.len()]))
        .collect()
}

fn register(value: &Vm) -> Option<u64> {
    match value {
        Vm::Number(n) => n
            .as_u64()
            .or_else(|| n.as_i64().filter(|x| *x >= 0).map(|x| x as u64))
            .or_else(|| {
                n.as_f64()
                    .filter(|x| {
                        x.is_finite() && *x >= 0.0 && x.fract() == 0.0 && *x < u64::MAX as f64
                    })
                    .map(|x| x as u64)
            }),
        _ => None,
    }
}

fn py_json(value: &Vm) -> Option<String> {
    match value {
        Vm::Null => Some("null".into()),
        Vm::Bool(v) => Some(if *v { "true" } else { "false" }.into()),
        Vm::Number(v) => Some(v.to_string()),
        Vm::String(v) => Some(py_json_string(v)),
        Vm::Array(values) => Some(format!(
            "[{}]",
            values
                .iter()
                .map(py_json)
                .collect::<Option<Vec<_>>>()?
                .join(", ")
        )),
        Vm::Object(fields) | Vm::OrderedMap(fields) => {
            let items = fields
                .iter()
                .map(|(k, v)| Some(format!("{}: {}", py_json_string(k), py_json(v)?)))
                .collect::<Option<Vec<_>>>()?;
            Some(format!("{{{}}}", items.join(", ")))
        }
        Vm::Function(_) => None,
    }
}
fn py_json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{0020}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c as u32 > 0x7f && c as u32 <= 0xffff => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c if c as u32 > 0xffff => {
                let n = c as u32 - 0x10000;
                out.push_str(&format!(
                    "\\u{:04x}\\u{:04x}",
                    0xd800 + (n >> 10),
                    0xdc00 + (n & 0x3ff)
                ));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct Interpreter {
    map: HashMap<u64, Vm>,
    result: String,
    started: Instant,
    key: String,
    steps: usize,
}
impl Interpreter {
    fn new(tokens: &Value, key: &str) -> Self {
        let mut map = HashMap::new();
        for opcode in [1, 2, 3, 5, 6, 7, 8, 14, 15, 17, 18, 19, 20, 21, 23, 24] {
            map.insert(opcode, Vm::Function(opcode));
        }
        map.insert(9, from_json(tokens));
        map.insert(10, Vm::String("window".into()));
        map.insert(16, Vm::String(key.into()));
        Self {
            map,
            result: String::new(),
            started: Instant::now(),
            key: key.into(),
            steps: 0,
        }
    }
    fn get_reg(&self, id: &Vm) -> Option<Vm> {
        self.map.get(&register(id)?).cloned()
    }
    fn put(&mut self, id: &Vm, value: Vm) {
        if let Some(id) = register(id) {
            self.map.insert(id, value);
        }
    }
    fn run(&mut self, opcode: u64, args: &[Vm], depth: usize) {
        self.steps += 1;
        if self.steps > MAX_INSTRUCTIONS || depth > MAX_RECURSION {
            return;
        }
        let arg = |index: usize| args.get(index);
        match opcode {
            // func_1(e,t): XOR the string form of registers e and t.
            1 if args.len() == 2 => {
                if let (Some(e), Some(t)) = (arg(0), arg(1)) {
                    if let (Some(x), Some(y)) = (self.get_reg(e), self.get_reg(t)) {
                        if let Some(value) = xor_string(&py_string(&x), &py_string(&y)) {
                            self.put(e, Vm::String(value));
                        }
                    }
                }
            }
            // func_2(e,t): t is a literal from the instruction, not a register.
            2 if args.len() == 2 => {
                if let Some(e) = arg(0) {
                    self.put(e, arg(1).cloned().unwrap_or(Vm::Null));
                }
            }
            // func_3(e): base64-encode the literal string argument.
            3 if args.len() == 1 => {
                if let Some(Vm::String(value)) = arg(0) {
                    if value.len() <= MAX_TEXT {
                        self.result = B64.encode(value.as_bytes());
                    }
                }
            }
            // func_5(e,t): append on arrays, otherwise use the source's string/float rules.
            5 if args.len() == 2 => {
                if let (Some(e), Some(t)) = (arg(0), arg(1)) {
                    if let (Some(current), Some(incoming)) = (self.get_reg(e), self.get_reg(t)) {
                        let next = match current {
                            Vm::Array(mut values) => {
                                values.push(incoming);
                                Vm::Array(values)
                            }
                            ref current
                                if is_string_or_float(current) || is_string_or_float(&incoming) =>
                            {
                                Vm::String(format!(
                                    "{}{}",
                                    py_string(current),
                                    py_string(&incoming)
                                ))
                            }
                            _ => Vm::String("NaN".into()),
                        };
                        self.put(e, next);
                    }
                }
            }
            6 if args.len() == 3 => self.dot(args),
            // func_7(e,*args): Reflect.set is the only callable built-in special case.
            7 if args.len() >= 1 => self.call_indirect(args, depth, false),
            8 if args.len() == 2 => {
                if let (Some(e), Some(t)) = (arg(0), arg(1)) {
                    if let Some(value) = self.get_reg(t) {
                        self.put(e, value);
                    }
                }
            }
            14 if args.len() == 2 => {
                if let (Some(e), Some(t)) = (arg(0), arg(1)) {
                    if let Some(Vm::String(serialized)) = self.get_reg(t) {
                        if serialized.len() <= MAX_TEXT {
                            if let Ok(value) = serde_json::from_str::<Value>(&serialized) {
                                self.put(e, from_json(&value));
                            }
                        }
                    }
                }
            }
            15 if args.len() == 2 => {
                if let (Some(e), Some(t)) = (arg(0), arg(1)) {
                    if let Some(value) = self.get_reg(t).and_then(|v| py_json(&v)) {
                        if value.len() <= MAX_TEXT {
                            self.put(e, Vm::String(value));
                        }
                    }
                }
            }
            17 if args.len() >= 2 => self.call_named(args, depth),
            18 if args.len() == 1 => {
                if let Some(e) = arg(0) {
                    if let Some(value) = self.get_reg(e) {
                        if let Ok(bytes) = B64.decode(py_string(&value)) {
                            if let Ok(text) = String::from_utf8(bytes) {
                                if text.len() <= MAX_TEXT {
                                    self.put(e, Vm::String(text));
                                }
                            }
                        }
                    }
                }
            }
            19 if args.len() == 1 => {
                if let Some(e) = arg(0) {
                    if let Some(value) = self.get_reg(e) {
                        let encoded = B64.encode(py_string(&value));
                        if encoded.len() <= MAX_TEXT {
                            self.put(e, Vm::String(encoded));
                        }
                    }
                }
            }
            20 if args.len() >= 3 => self.conditional_call(args, depth),
            21 => {}
            23 if args.len() >= 2 => self.call_direct(args, depth),
            24 if args.len() == 3 => self.dot(args),
            _ => {} // Python catches malformed and unknown instructions and continues.
        }
    }
    fn dot(&mut self, args: &[Vm]) {
        if let (Some(e), Some(t), Some(n)) = (args.first(), args.get(1), args.get(2)) {
            if let (Some(Vm::String(left)), Some(Vm::String(right))) =
                (self.get_reg(t), self.get_reg(n))
            {
                let joined = if left == "window.document" && right == "location" {
                    "https://chatgpt.com/".into()
                } else {
                    format!("{left}.{right}")
                };
                self.put(e, Vm::String(joined));
            }
        }
    }
    fn call_indirect(&mut self, args: &[Vm], depth: usize, _unused: bool) {
        let Some(target_reg) = args.first() else {
            return;
        };
        let Some(target) = self.get_reg(target_reg) else {
            return;
        };
        if matches!(&target, Vm::String(s) if s == "window.Reflect.set") {
            let values = args
                .iter()
                .skip(1)
                .filter_map(|id| self.get_reg(id))
                .collect::<Vec<_>>();
            if values.len() == 3 {
                self.set_property(values[0].clone(), py_string(&values[1]), values[2].clone());
            }
        } else if let Vm::Function(op) = target {
            let values = args
                .iter()
                .skip(1)
                .filter_map(|id| self.get_reg(id))
                .collect::<Vec<_>>();
            self.run(op, &values, depth + 1);
        }
    }
    fn call_named(&mut self, args: &[Vm], depth: usize) {
        let (Some(dest), Some(target_reg)) = (args.first(), args.get(1)) else {
            return;
        };
        let Some(target) = self.get_reg(target_reg) else {
            return;
        };
        let call_args = args
            .iter()
            .skip(2)
            .filter_map(|id| self.get_reg(id))
            .collect::<Vec<_>>();
        match target {
            Vm::String(name) if name == "window.performance.now" => self.put(
                dest,
                Vm::Number(
                    Number::from_f64(self.started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or_else(|| Number::from(0)),
                ),
            ),
            Vm::String(name) if name == "window.Object.create" => {
                self.put(dest, Vm::OrderedMap(Vec::new()))
            }
            Vm::String(name) if name == "window.Object.keys" => {
                if call_args
                    .first()
                    .is_some_and(|v| matches!(v, Vm::String(s) if s == "window.localStorage"))
                {
                    self.put(
                        dest,
                        Vm::Array(
                            [
                                "STATSIG_LOCAL_STORAGE_INTERNAL_STORE_V4",
                                "STATSIG_LOCAL_STORAGE_STABLE_ID",
                                "client-correlated-secret",
                                "oai/apps/capExpiresAt",
                                "oai-did",
                                "STATSIG_LOCAL_STORAGE_LOGGING_REQUEST",
                                "UiState.isNavigationCollapsed.1",
                            ]
                            .iter()
                            .map(|s| Vm::String((*s).into()))
                            .collect(),
                        ),
                    );
                }
            }
            Vm::String(name) if name == "window.Math.random" => {
                let n = (crate::random_u64() >> 11) as f64 / ((1u64 << 53) as f64);
                self.put(
                    dest,
                    Vm::Number(Number::from_f64(n).unwrap_or_else(|| Number::from(0))),
                );
            }
            Vm::Function(op) => {
                self.run(op, &call_args, depth + 1);
                self.put(dest, Vm::Null);
            }
            _ => {}
        }
    }
    fn conditional_call(&mut self, args: &[Vm], depth: usize) {
        let (Some(left), Some(right), Some(target_reg)) = (args.first(), args.get(1), args.get(2))
        else {
            return;
        };
        if self.get_reg(left) != self.get_reg(right) {
            return;
        }
        let Some(Vm::Function(op)) = self.get_reg(target_reg) else {
            return;
        };
        let call_args = args
            .iter()
            .skip(3)
            .filter_map(|id| self.get_reg(id))
            .collect::<Vec<_>>();
        self.run(op, &call_args, depth + 1);
    }
    fn call_direct(&mut self, args: &[Vm], depth: usize) {
        let (Some(receiver), Some(target_reg)) = (args.first(), args.get(1)) else {
            return;
        };
        if matches!(self.get_reg(receiver), Some(Vm::Null) | None) {
            return;
        }
        let Some(Vm::Function(op)) = self.get_reg(target_reg) else {
            return;
        };
        self.run(op, &args[2..], depth + 1);
    }
    fn set_property(&mut self, object: Vm, key: String, value: Vm) {
        let fields = match object {
            Vm::OrderedMap(fields) | Vm::Object(fields) => fields,
            _ => return,
        };
        let mut fields = fields;
        if let Some((_, old)) = fields.iter_mut().find(|(name, _)| name == &key) {
            *old = value;
        } else if fields.len() < 1024 {
            fields.push((key, value));
        }
        // OrderedMap is a reference in Python; update every register that refers to it.
        for candidate in self.map.values_mut() {
            if matches!(candidate, Vm::OrderedMap(_) | Vm::Object(_)) {
                *candidate = Vm::OrderedMap(fields.clone());
            }
        }
    }
}
fn is_string_or_float(value: &Vm) -> bool {
    matches!(value, Vm::String(_)) || matches!(value, Vm::Number(n) if n.is_f64())
}

/// Solve a `dx` challenge using the exact instruction subset in the reference VM.
pub fn solve(dx: &str, key: &str) -> Option<String> {
    if dx.len() > MAX_PROGRAM_BYTES * 2 || key.len() > 4096 {
        return None;
    }
    let decoded = B64.decode(dx).ok()?;
    if decoded.len() > MAX_PROGRAM_BYTES {
        return None;
    }
    let encoded_text = String::from_utf8(decoded).ok()?;
    let plaintext = xor_string(&encoded_text, key)?;
    let tokens: Value = serde_json::from_str(&plaintext).ok()?;
    let instructions = tokens.as_array()?;
    if instructions.len() > MAX_INSTRUCTIONS {
        return None;
    }
    let mut vm = Interpreter::new(&tokens, key);
    for instruction in instructions {
        let Some(parts) = instruction.as_array() else {
            continue;
        };
        let Some(opcode) = parts.first().and_then(|v| register(&from_json(v))) else {
            continue;
        };
        let args = parts.iter().skip(1).map(from_json).collect::<Vec<_>>();
        vm.run(opcode, &args, 0);
    }
    if vm.result.is_empty() {
        None
    } else {
        Some(vm.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_program_calls_only_allowlisted_function() {
        let program = serde_json::json!([
            [2, 50, "hello"],
            [2, 51, "window"],
            [5, 50, 51],
            [8, 60, 3],
            [7, 60, 50]
        ]);
        let key = "turnstile-test";
        let encoded = B64.encode(xor_string(&program.to_string(), key).unwrap());
        assert_eq!(solve(&encoded, key), Some(B64.encode(b"hellowindow")));
    }
    #[test]
    fn rejects_non_json_program_and_oversize_payload() {
        assert_eq!(solve(&B64.encode(b"eval('1+1')"), "p"), None);
        assert_eq!(solve(&"A".repeat(MAX_PROGRAM_BYTES * 2 + 1), "p"), None);
    }
    #[test]
    fn python_json_dumps_ascii_and_spaces() {
        let value = Vm::Object(vec![("word".into(), Vm::String("café".into()))]);
        assert_eq!(
            py_json(&value).as_deref(),
            Some("{\"word\": \"caf\\u00e9\"}")
        );
    }
}
