//! Released structural later-system preparation, performed without Python.
//! AST and token offsets come from the pinned MiniJinja parser, never text search.
use crate::{bad, chat_templates::{Template, Templates}, Frontend, Result};
use minijinja::machinery::{ast::{self, Expr, Stmt}, parse, tokenize, Token};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

const MARKER: &str = "JustFit later-system canary 5d0c8e";
#[derive(PartialEq)]
enum Original { Renders, Rejects, Drops, Misplaces }

pub fn defined(path: &Path) -> Result<Templates> {
    // transformers gives chat_template.jinja precedence over tokenizer config;
    // named templates may instead be stored in chat_templates/*.jinja.
    let single = path.join("chat_template.jinja");
    if single.is_file() { return Ok(Templates::raw(std::fs::read_to_string(single).map_err(|e| bad(e.to_string()))?)); }
    let named = path.join("chat_templates");
    let mut entries = BTreeMap::new();
    if named.is_dir() {
        for entry in std::fs::read_dir(named).map_err(|e| bad(e.to_string()))? {
            let entry = entry.map_err(|e| bad(e.to_string()))?;
            let path = entry.path();
            if path.extension().is_some_and(|s| s == "jinja") {
                let name = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| bad("invalid named chat template"))?.to_string();
                entries.insert(name, Template { source: std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?, later_system: "unsupported".into() });
            }
        }
    }
    if entries.is_empty() {
        let config = path.join("tokenizer_config.json");
        let value: Value = serde_json::from_slice(&std::fs::read(config).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
        match &value["chat_template"] {
            Value::String(source) => return Ok(Templates::raw(source.clone())),
            Value::Object(named) => for (name, source) in named { entries.insert(name.clone(), entry(source)?); },
            Value::Array(named) => for value in named {
                let name = value["name"].as_str().ok_or_else(|| bad("invalid named chat template"))?;
                entries.insert(name.to_string(), entry(&value["template"])?);
            },
            _ => return Err(bad("the tokenizer defines no chat template")),
        }
    }
    if !entries.contains_key("default") { return Err(bad("the tokenizer's named chat templates have no default")); }
    let templates = Templates { named: true, entries };
    templates.validate()?;
    Ok(templates)
}
fn entry(source: &Value) -> Result<Template> {
    let source = source.as_str().filter(|s| !s.is_empty()).ok_or_else(|| bad("the tokenizer defines no chat template"))?;
    Ok(Template { source: source.into(), later_system: "unsupported".into() })
}

fn in_place(text: &str) -> bool {
    let (before, marker, after) = (text.find("First answer"), text.find(MARKER), text.find("Next question"));
    text.matches(MARKER).count() == 1 && matches!((before, marker, after), (Some(a),Some(b),Some(c)) if a+"First answer".len()<=b && b+MARKER.len()<=c)
}
fn canaries() -> Vec<(Value, Value)> {
    [true, false].into_iter().map(|leading| {
        let mut messages = vec![];
        if leading { messages.push(json!({"role":"system","content":"Leading instructions"})); }
        messages.extend([json!({"role":"user","content":"First question"}),json!({"role":"assistant","content":"First answer"}),json!({"role":"user","content":"Next question"})]);
        let without = json!(messages);
        messages.insert(messages.len()-1, json!({"role":"system","content":MARKER}));
        (json!(messages), without)
    }).collect()
}
fn original(front: &Frontend, source: &str) -> Original {
    for (canary, _) in canaries() {
        let Ok(text) = front.render_messages(&json!({"add_generation_prompt":true}),canary,source) else { return Original::Rejects };
        if !text.contains(MARKER) { return Original::Drops; }
        if !in_place(&text) { return Original::Misplaces; }
    }
    Original::Renders
}

fn ordinary() -> Vec<Value> {
    let system=json!({"role":"system","content":"Leading instructions"});
    let ask=json!({"role":"user","content":"First question"});
    let thought=json!({"role":"assistant","content":"First answer","reasoning_content":"Hm"});
    let next=json!({"role":"user","content":"Next question"});
    let call=json!({"role":"assistant","content":"","reasoning_content":"Look it up","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"key":"alpha"}}}]});
    let result=json!({"role":"tool","tool_call_id":"call_1","content":"beta"});
    let image=json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}},{"type":"text","text":"Describe it"}]});
    vec![json!([ask]),json!([system,ask]),json!([system,ask,thought,next]),json!([system,ask,call,result]),json!([system,ask,call,result,thought,next]),json!([system,image])]
}
fn options() -> Vec<Value> {
    let tools=json!([{"type":"function","function":{"name":"lookup","description":"Look up a key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}}]);
    let mut options=vec![];
    for with_tools in [false,true] { for effort in [None,Some("none"),Some("minimal"),Some("low"),Some("medium"),Some("high"),Some("xhigh"),Some("max")] { for preserve in [None,Some(true),Some(false)] {
        let mut body=json!({"add_generation_prompt":true});
        if with_tools { body["tools"]=tools.clone(); }
        if let Some(effort)=effort { body["reasoning_effort"]=json!(effort); }
        if let Some(preserve)=preserve { body["preserve_thinking"]=json!(preserve); }
        options.push(body);
    }}}
    options.push(json!({"add_generation_prompt":false}));
    options
}
fn verified(front: &Frontend, source: &str, patched: &str) -> bool {
    let ordinary=ordinary(); let canaries=canaries();
    for options in options() {
        for messages in &ordinary {
            let original=front.render_messages(&options,messages.clone(),source);
            let patched=front.render_messages(&options,messages.clone(),patched);
            // Both errors have the same released rendering error at this layer.
            match (original,patched) { (Ok(a),Ok(b)) if a==b => (), (Err(a),Err(b)) if a.0==b.0 && a.1==b.1 => (), _=>return false }
        }
        for (canary,without) in &canaries {
            if front.render_messages(&options,without.clone(),source).is_err() {continue;}
            if !front.render_messages(&options,canary.clone(),patched).is_ok_and(|text|in_place(&text)) {return false;}
        }
    }
    true
}

fn system_block(front: &Frontend, source: &str) -> Option<(String,String)> {
    let body=json!({"reasoning_effort":"none","add_generation_prompt":false});
    let with=front.render_messages(&body,json!([{"role":"system","content":MARKER},{"role":"user","content":"First question"}]),source).ok()?;
    let without=front.render_messages(&body,json!([{"role":"user","content":"First question"}]),source).ok()?;
    let block=with.strip_suffix(&without)?;
    if block.matches(MARKER).count()!=1 {return None;}
    let (before,after)=block.split_once(MARKER)?;
    Some((before.into(),after.into()))
}

// The pinned parser exposes node spans; tokenize supplies full tag boundaries,
// ignoring strings, comments and raw sections exactly as the renderer does.
fn tags(source: &str) -> Option<Vec<(usize,usize)>> {
    let mut result=vec![]; let mut start=None;
    for token in tokenize(source,false,Default::default()) {
        let (token,span)=token.ok()?;
        match token {
            Token::VariableStart|Token::BlockStart=>start=Some(span.start_offset as usize),
            Token::VariableEnd|Token::BlockEnd=>result.push((start.take()?,span.end_offset as usize)),
            _=>(),
        }
    }
    Some(result)
}
fn var(expr: &Expr<'_>, name: &str) -> bool { matches!(expr,Expr::Var(v) if v.id==name) }
fn role(expr: &Expr<'_>, variable: &str) -> bool {
    match expr {
        Expr::GetAttr(v)=>v.name=="role" && var(&v.expr,variable),
        Expr::GetItem(v)=>var(&v.expr,variable) && matches!(&v.subscript_expr,Expr::Const(c) if c.value.as_str()==Some("role")),
        _=>false,
    }
}
fn compares(expr: &Expr<'_>, variable: &str, equal: bool) -> bool {
    match expr {
        Expr::Compare(v)=>v.ops.len()==1 && role(&v.expr,variable) && matches!((&v.ops[0].op,equal),(ast::CompareOpKind::Eq,true)|(ast::CompareOpKind::Ne,false)) && matches!(&v.ops[0].expr,Expr::Const(c) if c.value.as_str()==Some("system")),
        Expr::BinOp(v)=>role(&v.left,variable) && matches!((&v.op,equal),(ast::BinOpKind::Eq,true)|(ast::BinOpKind::Ne,false)) && matches!(&v.right,Expr::Const(c) if c.value.as_str()==Some("system")),
        _=>false,
    }
}
fn has_term(expr: &Expr<'_>, variable: &str, equal: bool) -> bool {
    if let Expr::BinOp(v)=expr {
        if matches!((&v.op,equal),(ast::BinOpKind::ScOr,true)|(ast::BinOpKind::ScAnd,false)) {return has_term(&v.left,variable,equal)||has_term(&v.right,variable,equal);}
    }
    compares(expr,variable,equal)
}
fn children<'a,'s>(stmt: &'a Stmt<'s>) -> Vec<&'a [Stmt<'s>]> {
    match stmt {
        Stmt::Template(v)=>vec![&v.children],
        Stmt::ForLoop(v)=>vec![&v.body,&v.else_body],
        Stmt::IfCond(v)=>vec![&v.true_body,&v.false_body],
        Stmt::WithBlock(v)=>vec![&v.body],
        Stmt::SetBlock(v)=>vec![&v.body],
        Stmt::AutoEscape(v)=>vec![&v.body],
        Stmt::FilterBlock(v)=>vec![&v.body],
        Stmt::Block(v)=>vec![&v.body],
        Stmt::Macro(v)=>vec![&v.body],
        Stmt::CallBlock(v)=>vec![&v.macro_decl.body],
        _=>vec![],
    }
}
fn visit<'a,'s>(body: &'a [Stmt<'s>], callback: &mut impl FnMut(&'a Stmt<'s>)) {
    for stmt in body {callback(stmt);for body in children(stmt) {visit(body,callback);}}
}
fn construct(source: &str, rejecting: bool) -> Option<((usize,usize),String)> {
    let tree=parse(source,"chat",Default::default()).ok()?;
    let tags=tags(source)?;
    let mut found=BTreeMap::new();
    visit(std::slice::from_ref(&tree),&mut |stmt| {
        let Stmt::ForLoop(loop_)=stmt else {return};
        let Expr::Var(variable)=&loop_.target else {return};
        if !var(&loop_.iter,"messages")||loop_.filter_expr.is_some() {return;}
        let mut remember=|offset: usize| {if let Some(span)=tags.iter().find(|(start,end)|*start<=offset && offset<*end) {found.insert(*span,variable.id.to_string());}};
        if rejecting {
            visit(&loop_.body,&mut |stmt| {
                let Stmt::IfCond(branch)=stmt else{return};
                if !has_term(&branch.expr,variable.id,true) {return;}
                visit(&branch.true_body,&mut |stmt| {
                    if let Stmt::EmitExpr(output)=stmt {
                        if matches!(&output.expr,Expr::Call(call) if var(&call.expr,"raise_exception")) {remember(output.span().start_offset as usize);}
                    }
                });
            });
        } else {
            for stmt in &loop_.body {
                if let Stmt::IfCond(branch)=stmt {
                    if branch.false_body.is_empty() && has_term(&branch.expr,variable.id,false) {remember(branch.span().start_offset as usize);}
                }
            }
        }
    });
    if found.len()!=1 {return None;}
    found.into_iter().next()
}
fn literal(text: &str) -> String {
    // Match Python unicode_escape so patch source itself remains reproducible.
    let mut out=String::from("'");
    for ch in text.chars() {
        match ch { '\''=>out.push_str("\\'"),'\\'=>out.push_str("\\\\"),'\n'=>out.push_str("\\n"),'\r'=>out.push_str("\\r"),'\t'=>out.push_str("\\t"),
            ch if (' '..='~').contains(&ch)=>out.push(ch),
            ch=>{use std::fmt::Write;let n=ch as u32;if n<=255 {write!(out,"\\x{n:02x}").unwrap();}else if n<=65535 {write!(out,"\\u{n:04x}").unwrap();}else {write!(out,"\\U{n:08x}").unwrap();}}
        }
    }
    out.push('\'');out
}
fn patch(front: &Frontend, source: &str, rejecting: bool) -> Option<String> {
    let source=source.replace("\r\n","\n").replace('\r',"\n");
    let ((start,end),variable)=construct(&source,rejecting)?;
    let (prefix,suffix)=system_block(front,&source)?;
    let later=format!("{{{{- {} + ({variable}.content|trim) + {} -}}}}",literal(&prefix),literal(&suffix));
    let replacement=if rejecting {later} else {
        let tag=&source[start..end];
        let mut head=2;
        if tag.as_bytes().get(head).is_some_and(|v|matches!(v,b'-'|b'+')) {head+=1;}
        while tag.as_bytes().get(head).is_some_and(u8::is_ascii_whitespace) {head+=1;}
        format!("{}if not loop.first and {variable}.role == 'system' %}}{later}{}el{}",&tag[..head],&tag[..head],&tag[head..])
    };
    Some(format!("{}{}{}",&source[..start],replacement,&source[end..]))
}
pub fn prepare(front: &Frontend, mut templates: Templates) -> Result<Templates> {
    templates.validate()?;
    let mut prepared=BTreeMap::<String,Template>::new();
    for template in templates.entries.values_mut() {
        let source=template.source.clone();
        if let Some(found)=prepared.get(&source) {*template=found.clone();continue;}
        let original=original(front,&source);
        if original==Original::Renders {template.later_system="native".into();}
        else if original==Original::Rejects || original==Original::Drops {
            if let Some(patched)=patch(front,&source,original==Original::Rejects) {
                if verified(front,&source,&patched) {template.source=patched;template.later_system="patched".into();}
            }
        }
        prepared.insert(source,template.clone());
    }
    Ok(templates)
}
