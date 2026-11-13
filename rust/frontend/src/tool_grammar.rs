//! Tool-choice grammar construction; parameter grammars remain request-local.
use crate::{bad,tool_arguments::{self,with_root},tools::{lookup,schema_types,Policy},Result};
use serde_json::{json,Value};
use std::collections::HashMap;
const END:&str="\n</parameter>\n";
const NAME:&str=r"/[^<> \t\n\r]([^<>\n\r]{0,254}[^<> \t\n\r])?/";
pub const WS:&str=r"WS: /[ \t\n\r]{0,64}/";
fn quoted(text:&str)->String{json!(text).to_string()}
fn literal_texts(schema:&Value,root:&Value)->Option<Vec<String>> {
    fn read(node:&Value,root:&Value,memo:&mut HashMap<usize,Option<Vec<String>>>)->Option<Vec<String>> {
        let key=node as *const Value as usize;if let Some(value)=memo.get(&key){return value.clone();}memo.insert(key,None);
        let map=node.as_object()?;
        let values=map.get("const").map(|v|vec![v]).or_else(||map.get("enum").and_then(Value::as_array).map(|a|a.iter().collect()));
        let kind=map.get("type").and_then(|v|v.as_str().or_else(||v.as_array().filter(|a|a.len()==1).and_then(|a|a[0].as_str())));
        let members=map.get("anyOf").or_else(||map.get("oneOf")).and_then(Value::as_array).filter(|a|!a.is_empty());
        let mut result=if let Some(values)=values {Some(values.iter().map(|v|v.as_str().map(str::to_string).unwrap_or_else(||v.to_string())).collect::<Vec<_>>())}
        else if kind==Some("null"){Some(vec!["null".into()])}else if kind==Some("boolean"){Some(vec!["true".into(),"false".into()])}
        else if let Some(members)=members {let mut result=Some(Vec::new());for member in members {if let Some(texts)=read(member,root,memo){result.as_mut()?.extend(texts);}else{result=None;break;}}result}
        else if let Some(target)=map.get("$ref").and_then(Value::as_str).and_then(|r|lookup(r,root).ok()){read(target,root,memo)}else{None};
        for member in map.get("allOf").and_then(Value::as_array).into_iter().flatten(){if let Some(other)=read(member,root,memo){result=Some(match result {None=>other,Some(values)=>values.into_iter().filter(|s|other.contains(s)).collect()});}}
        if let Some(values)=&mut result{let mut unique=Vec::new();for value in std::mem::take(values){if !unique.contains(&value){unique.push(value);}}*values=unique;}
        memo.insert(key,result.clone());result
    }
    read(schema,root,&mut HashMap::new())
}
fn members(schema:&Value)->Vec<&Value> {
    let annotations=["$comment","title","description","default","examples","deprecated","readOnly","writeOnly","$schema","$id","$anchor","$dynamicAnchor","$defs","definitions","anyOf","oneOf"];
    if let Some(map)=schema.as_object(){if map.keys().all(|k|annotations.contains(&k.as_str())){if let Some(array)=map.get("anyOf").or_else(||map.get("oneOf")).and_then(Value::as_array).filter(|a|!a.is_empty()){return array.iter().collect();}}}
    vec![schema]
}
fn parameter(rule:&str,prefix:&str,schema:&Value,compatible:&mut impl FnMut(Value)->Result<Value>)->Result<Option<Vec<String>>> {
    if schema==false{return Ok(None);}let closing=quoted(END);let mut rules=Vec::new();let mut choices=Vec::new();
    for member in members(schema) {
        if let Some(texts)=literal_texts(member,schema) {
            for text in texts {
                if text.contains(END){return Err(bad("tool parameter enum contains XML framing"));}
                if text.is_empty(){let name=format!("{rule}_empty_{}",rules.len());rules.push(format!("{name}:"));choices.push(name);}else{choices.push(quoted(&text));}
            }
        }else{
            if schema_types(member,schema).is_none_or(|t|t.contains("string")){return Ok(Some(vec![format!("{rule}: {prefix} {rule}_value"),format!("{rule}_value[suffix={closing}]: /(?s:.*)/")]));}
            let member=if std::ptr::eq(member,schema){member.clone()}else{with_root(member,schema)};
            choices.push(format!("%json {}",compatible(member)?));
        }
    }
    if choices.is_empty(){return Ok(None);}let mut unique=Vec::new();for value in choices{if !unique.contains(&value){unique.push(value);}}
    rules.push(format!("{rule}: {prefix} ({}) {closing}",unique.join(" | ")));Ok(Some(rules))
}
pub fn arguments(schema:&Value,compatible:&mut impl FnMut(Value)->Result<Value>)->Result<String> {
    let properties=schema["properties"].as_object().ok_or_else(||bad("invalid framed tool schema"))?;let required=schema["required"].as_array().ok_or_else(||bad("invalid framed required list"))?;
    let mut rules=Vec::new();let mut sequence=Vec::new();
    for (index,(name,value)) in properties.iter().enumerate() {
        if value!=false && (name.is_empty()||name.chars().count()>256||name!=name.trim_matches([' ','\t','\n','\r'])||name.contains(['<','>','\n','\r'])){return Err(bad("invalid tool parameter name"));}
        let rule=format!("parameter_{index}");let prefix=format!("{} {} {}",quoted("<parameter="),quoted(name),quoted(">\n"));
        if let Some(next)=parameter(&rule,&prefix,value,compatible)?{rules.extend(next);sequence.push(rule+if required.contains(&json!(name)){""}else{"?"});}
        else if required.contains(&json!(name)){return Err(bad(format!("required tool parameter cannot have a value: {name}")));}
    }
    let prefix=format!("{} EXTRA_NAME {}",quoted("<parameter="),quoted(">\n"));
    if let Some(extra)=parameter("extra",&prefix,&schema["additionalProperties"],compatible)? {
        let declared=properties.keys().map(|name|quoted(name)).collect::<Vec<_>>().join(" | ");rules.push(format!("EXTRA_NAME: {NAME}{}",if declared.is_empty(){String::new()}else{format!(" & ~({declared})")}));rules.extend(extra);sequence.push("extra*".into());
    }
    Ok(format!("%llguidance {{}}\nstart:{}\n{}\n",sequence.iter().map(|s|format!(" {s}")).collect::<String>(),rules.join("\n")))
}
pub fn build(policy:&Policy,thinking:bool,response:Option<&Value>,compatible:&mut impl FnMut(Value)->Result<Value>)->Result<Value> {
    let mut budget=tool_arguments::BUDGET;let mut strict=HashMap::new();
    for name in &policy.strict {let root=tool_arguments::strict_schema(&policy.schemas[name]);let framed=tool_arguments::frame(&root,&mut budget)?;strict.insert(name,arguments(&framed,compatible)?);}
    let mut sides=Vec::new();if strict.len()<policy.schemas.len(){sides.push(json!({"name":"free_arguments","lark_grammar":format!("%llguidance {{}}\nstart: parameter*\nparameter: {} NAME {} value\nvalue[suffix={}]: /(?s:.*)/\nNAME: {NAME}\n",quoted("<parameter="),quoted(">\n"),quoted(END))}));}
    let mut calls=Vec::new();
    for (index,name) in policy.schemas.keys().enumerate(){let arguments=if let Some(grammar)=strict.get(name){let key=format!("arguments_{index}");sides.push(json!({"name":key,"lark_grammar":grammar}));key}else{"free_arguments".into()};calls.push(format!("call_{index}: <tool_call> {} @{arguments} {} </tool_call>",quoted(&format!("\n<function={name}>\n")),quoted("</function>\n")));}
    let call=format!("({})",(0..calls.len()).map(|i|format!("call_{i}")).collect::<Vec<_>>().join(" | "));
    let body=if calls.is_empty(){if response.is_none(){"TEXT".into()}else{"answer".into()}}
    else if policy.required||response.is_some(){let body=format!("WS {call} {}",if policy.parallel{format!("(TEXT {call})* TEXT")}else{"WS".into()});if policy.required{body}else{format!("(({body}) | answer)")}}
    else if policy.parallel{format!("(TEXT {call})* TEXT")}else{format!("TEXT ({call} WS)?")};
    let mut main=vec!["%llguidance {}".into(),format!("start: {}{body}",if thinking{"think "}else{""})];
    if !policy.required{if let Some(schema)=response{main.push(format!("answer: WS %json {} WS",compatible(schema.clone())?));}}
    if thinking{main.push("think: TEXT <[248069]>".into());}main.extend(calls);main.push(WS.into());main.push(r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/".into());
    sides.insert(0,json!({"name":"tool_output","lark_grammar":main.join("\n")+"\n"}));Ok(json!({"grammars":sides}))
}
