//! Tool policy and argument type contracts; HTTP tool generation is separate.
use crate::{bad, grammar::schema_children, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

pub struct Policy {
    pub schemas: Map<String, Value>,
    pub required: bool,
    pub parallel: bool,
    pub strict: BTreeSet<String>,
    pub constrained: bool,
    pub namespaces: Value,
}
impl Policy {
    pub fn snapshot(&self) -> Value {
        json!({"schemas":self.schemas,"required":self.required,"parallel":self.parallel,"strict":self.strict,"constrained":self.constrained,"namespaces":self.namespaces})
    }
}
fn check_schema(schema: &Value, name: &str) -> Result<()> {
    let invalid=||bad(format!("invalid tool schema for {name}"));
    if !schema.is_boolean() && !schema.is_object() {return Err(invalid());}
    let mut pending=vec![(schema,1)];
    while let Some((value,depth))=pending.pop() {
        if (value.is_object()||value.is_array())&&depth>64 {return Err(bad(format!("invalid tool schema for {name}: nested more than 64 levels deep")));}
        match value {Value::Object(map)=>pending.extend(map.values().map(|v|(v,depth+1))),Value::Array(array)=>pending.extend(array.iter().map(|v|(v,depth+1))),_=>{}}
    }
    let mut copy=schema.clone();
    schema_children(&mut copy,&mut |value| {
        if let Some(map)=value.as_object() {
            if map.get("$schema").is_some_and(|v|!v.is_string()){return Err(bad("$schema must be a string"));}
            for key in ["$ref","$dynamicRef","$recursiveRef"] {
                if let Some(reference)=map.get(key) {
                    let reference=reference.as_str().ok_or_else(||bad(format!("{key} must be a string")))?;
                    if !reference.starts_with('#'){return Err(bad(format!("remote tool schema reference is not allowed: {reference}")));}
                }
            }
            static PATTERN:OnceLock<jsonschema::Validator>=OnceLock::new();
            let pattern_validator=PATTERN.get_or_init(||jsonschema::options().should_validate_formats(true).build(&json!({"type":"string","format":"regex"})).expect("constant regex format schema"));
            if map.get("pattern").and_then(Value::as_str).is_some_and(|pattern|!pattern_validator.is_valid(&json!(pattern))) {
                return Err(invalid());
            }
            if map.get("patternProperties").and_then(Value::as_object).is_some_and(|patterns|patterns.keys().any(|pattern|!pattern_validator.is_valid(&json!(pattern)))) {
                return Err(invalid());
            }
        }
        Ok(())
    })?;
    // Meta-validation deliberately does not resolve an unstrict tool's local
    // references; Python only requires resolution for strict tools.
    match jsonschema::meta::try_validate(schema) {
        Ok(Ok(()))=>Ok(()),
        Ok(Err(error))=>Err(bad(format!("invalid tool schema for {name}: {error}"))),
        Err(_)=>Err(invalid()),
    }
}
pub(crate) fn children(value: &Value) -> Vec<&Value> {
    let mut output=Vec::new();
    if let Some(map)=value.as_object() {
        for (key,value) in map {
            if ["properties","patternProperties","$defs","definitions","dependentSchemas","dependencies"].contains(&key.as_str()) {
                if let Some(map)=value.as_object(){output.extend(map.values().filter(|v|key!="dependencies"||!v.is_array()));}
            } else if ["items","prefixItems","additionalItems","contains","additionalProperties","unevaluatedItems","unevaluatedProperties","propertyNames","allOf","anyOf","oneOf","not","if","then","else","contentSchema","extends"].contains(&key.as_str()) {
                match value {Value::Array(array)=>output.extend(array),_=>output.push(value)}
            } else if ["type","disallow"].contains(&key.as_str()) {
                if let Some(array)=value.as_array(){output.extend(array.iter().filter(|v|v.is_object()));}
            }
        }
    }
    output
}
pub(crate) fn lookup<'a>(reference: &str, root: &'a Value) -> Result<&'a Value> {
    let error=||bad(format!("unresolved tool parameter reference: {reference}"));
    let fragment=reference.strip_prefix('#').ok_or_else(||bad("unsupported tool parameter reference"))?;
    let fragment=percent_encoding::percent_decode_str(fragment).decode_utf8_lossy();
    if fragment.is_empty(){return Ok(root);}
    if let Some(pointer)=fragment.strip_prefix('/') {
        let mut value=root;
        for part in pointer.split('/') {
            let part=part.replace("~1","/").replace("~0","~");
            value=match value {
                Value::Object(map)=>map.get(&part).ok_or_else(error)?,
                Value::Array(array) if part=="0" || (!part.is_empty()&&!part.starts_with('0')&&part.bytes().all(|b|b.is_ascii_digit()))=>array.get(part.parse::<usize>().map_err(|_|error())?).ok_or_else(error)?,
                _=>return Err(error()),
            };
        }
        return Ok(value);
    }
    let mut pending=vec![root];
    while let Some(value)=pending.pop() {
        if value.get("$anchor").and_then(Value::as_str)==Some(&fragment) || value.get("$dynamicAnchor").and_then(Value::as_str)==Some(&fragment){return Ok(value);}
        pending.extend(children(value));
    }
    Err(error())
}
pub fn normalize(body: &Value) -> Result<(Option<Value>,Option<Policy>)> {
    let parallel=body.get("parallel_tool_calls").filter(|v|!v.is_null()).map_or(Ok(true),|v|v.as_bool().ok_or_else(||bad("parallel_tool_calls must be a boolean")))?;
    let tools=&body["tools"];let choice=body.get("tool_choice").filter(|v|!v.is_null()).cloned().unwrap_or(json!("auto"));
    if tools.is_null(){if choice!="auto"&&choice!="none"{return Err(bad("tool_choice requires tools"));}return Ok((None,None));}
    let array=tools.as_array().ok_or_else(||bad("tools must be an array"))?;let mut schemas=Map::new();let mut marked=BTreeSet::new();
    for tool in array {
        if !tool.is_object() || tool["type"]!="function"{return Err(bad("only function tools are supported"));}
        let function=&tool["function"];let name=function["name"].as_str().filter(|name|!name.is_empty()&&name.len()<=128&&name.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_'||b==b'-')).ok_or_else(||bad("tool name must match [A-Za-z0-9_-]{1,128}"))?;
        if schemas.contains_key(name){return Err(bad(format!("duplicate tool name: {name}")));}
        let schema=function.get("parameters").filter(|v|!v.is_null()).cloned().unwrap_or(json!({}));
        if function.get("strict").is_some_and(|v|!v.is_null()&&!v.is_boolean()){return Err(bad(format!("strict must be a boolean for tool {name}")));}
        check_schema(&schema,name)?;schemas.insert(name.to_string(),schema);if function["strict"]==true{marked.insert(name.to_string());}
    }
    if array.is_empty(){if choice!="auto"&&choice!="none"{return Err(bad("tool_choice requires at least one tool"));}return Ok((None,None));}
    let namespaces=body.get("namespaces").filter(|v|!v.is_null()).cloned().unwrap_or(json!({}));
    if choice=="none"{return Ok((Some(tools.clone()),Some(Policy{schemas:Map::new(),required:false,parallel,strict:BTreeSet::new(),constrained:true,namespaces})));}
    let mut parallel=parallel;
    if choice.is_object() {
        let name=choice["function"]["name"].as_str().ok_or_else(||bad("invalid named tool_choice"))?;
        if choice["type"]!="function"||!schemas.contains_key(name){return Err(bad("invalid named tool_choice"));}
        let schema=schemas[name].clone();schemas.clear();schemas.insert(name.to_string(),schema);parallel=false;
    } else if choice!="auto"&&choice!="required"{return Err(bad("invalid tool_choice"));}
    let strict=marked.into_iter().filter(|name|schemas.contains_key(name)).collect::<BTreeSet<_>>();
    for name in &strict {
        let root=&schemas[name];let mut pending=vec![root];
        while let Some(value)=pending.pop(){if let Some(reference)=value.get("$ref").and_then(Value::as_str){lookup(reference,root)?;}pending.extend(children(value));}
    }
    let required=choice!="auto";let constrained=required||!parallel||!strict.is_empty();
    Ok((Some(tools.clone()),Some(Policy{schemas,required,parallel,strict,constrained,namespaces})))
}
pub fn fixture(body: &Value) -> Result<Value> {
    let (tools,policy)=normalize(body)?;Ok(json!({"tools":tools,"policy":policy.map(|v|v.snapshot())}))
}

type Types = Option<BTreeSet<String>>;
fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null", Value::Bool(_) => "boolean", Value::String(_) => "string",
        Value::Array(_) => "array", Value::Object(_) => "object",
        Value::Number(number) if !number.to_string().contains(['.','e','E']) => "integer",
        Value::Number(_) => "number",
    }
}
fn narrowed(left: Types, right: Types) -> Types {
    match (left,right) {
        (None,value)|(value,None)=>value,
        (Some(left),Some(right))=> {
            let mut both=left.intersection(&right).cloned().collect::<BTreeSet<_>>();
            if (left.contains("number")&&right.contains("integer")) || (left.contains("integer")&&right.contains("number")){both.insert("integer".into());}
            Some(both)
        }
    }
}
/// Match Python's conversion contract, including cyclic local references.
/// `None` admits any type; an empty set admits none.
pub fn schema_types(schema: &Value, root: &Value) -> Types {
    fn read(node:&Value,root:&Value,memo:&mut HashMap<usize,Types>)->Types {
        let Some(map)=node.as_object() else {return if node==false {Some(BTreeSet::new())}else{None};};
        let key=node as *const Value as usize;
        if let Some(value)=memo.get(&key){return value.clone();}
        memo.insert(key,None);
        let kind=map.get("type");
        let enumeration=map.get("const").map(|v|vec![v]).or_else(||map.get("enum").and_then(Value::as_array).map(|a|a.iter().collect()));
        let members=map.get("anyOf").or_else(||map.get("oneOf")).and_then(Value::as_array);
        let mut result=match kind {
            Some(Value::String(kind))=>Some(BTreeSet::from([kind.clone()])),
            Some(Value::Array(kinds))=>Some(kinds.iter().filter_map(Value::as_str).map(str::to_string).collect()),
            _=>if let Some(values)=enumeration {Some(values.into_iter().map(|v|value_kind(v).to_string()).collect())}
               else if let Some(members)=members.filter(|a|!a.is_empty()) {
                   let known=members.iter().filter_map(|v|read(v,root,memo)).collect::<Vec<_>>();
                   if known.is_empty(){None}else{Some(known.into_iter().flatten().collect())}
               }else{None},
        };
        if let Some(members)=map.get("allOf").and_then(Value::as_array){for member in members {result=narrowed(result,read(member,root,memo));}}
        if let Some(target)=map.get("$ref").and_then(Value::as_str).and_then(|reference|lookup(reference,root).ok()){result=narrowed(result,read(target,root,memo));}
        memo.insert(key,result.clone());result
    }
    read(schema,root,&mut HashMap::new())
}
pub fn types_fixture(body:&Value)->Value {
    json!(schema_types(&body["schema"],body.get("root").unwrap_or(&body["schema"])))
}
/// Decode XML parameter text using the declared conversion types. This does
/// not validate the value against its tool schema; final validation is separate.
pub fn convert_value(text:&str,types:Option<&BTreeSet<String>>)->Value {
    let original=||json!(text);
    let value=crate::strict_json(text.as_bytes()).ok().filter(|value| {
        let mut pending=vec![(value,2)];
        while let Some((value,depth))=pending.pop() {
            match value {
                Value::Object(map)=>{if depth>123{return false;}pending.extend(map.values().map(|v|(v,depth+1)));},
                Value::Array(array)=>{if depth>123{return false;}pending.extend(array.iter().map(|v|(v,depth+1)));},
                _=>{}
            }
        }
        true
    });
    let Some(value)=value else {
        let literal=match text.trim_matches(|c:char|c.is_whitespace()||('\u{1c}'..='\u{1f}').contains(&c)) {
            "True"=>Some(json!(true)),"False"=>Some(json!(false)),"None"=>Some(Value::Null),_=>None,
        };
        return match (literal,types) {
            (Some(value),Some(types)) if !types.contains("string")&&types.contains(value_kind(&value))=>value,
            _=>original(),
        };
    };
    if types.is_none_or(|types|types.contains("string")) {
        let kind=value_kind(&value);
        let declared=types.is_none_or(|types|types.contains(kind)||(kind=="integer"&&types.contains("number")));
        if declared&&kind!="string"{value}else{original()}
    }else{value}
}
pub fn conversion_fixture(body:&Value)->Result<Value> {
    let text=body["text"].as_str().ok_or_else(||bad("parameter text must be a string"))?;
    let types=body.get("types").filter(|v|!v.is_null()).map(|v|v.as_array().ok_or_else(||bad("types must be an array"))?.iter().map(|v|v.as_str().map(str::to_string).ok_or_else(||bad("type must be a string"))).collect::<Result<BTreeSet<_>>>()).transpose()?;
    Ok(convert_value(text,types.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recursive_types_and_numeric_intersections_match_conversion_contract() {
        let root=json!({"$defs":{"a":{"anyOf":[{"$ref":"#/$defs/a"},{"type":"integer"}]}}});
        assert_eq!(schema_types(&json!({"$ref":"#/$defs/a"}),&root),Some(BTreeSet::from(["integer".into()])));
        let schema=json!({"type":"number","allOf":[{"type":["integer","string"]}]});
        assert_eq!(schema_types(&schema,&schema),Some(BTreeSet::from(["integer".into()])));
        assert_eq!(schema_types(&Value::Bool(false),&root),Some(BTreeSet::new()));
        assert_eq!(schema_types(&json!({"$ref":"#missing"}),&root),None);
    }
    #[test]
    fn arguments_preserve_large_integers_and_string_spelling() {
        let number="18446744073709551617";
        assert_eq!(convert_value(number,None).to_string(),number);
        assert_eq!(convert_value("\"hello\"",None),json!("\"hello\""));
        let boolean=BTreeSet::from(["boolean".into()]);
        assert_eq!(convert_value("\u{1c}True\u{1f}",Some(&boolean)),json!(true));
        assert_eq!(convert_value("1e999",None),json!("1e999"));
    }
    #[test]
    fn strict_references_are_required_but_unstrict_unresolved_references_are_allowed() {
        let mut body=json!({"tools":[{"type":"function","function":{"name":"lookup","parameters":{"$ref":"#/$defs/missing"}}}]});
        assert!(normalize(&body).is_ok());body["tools"][0]["function"]["strict"]=json!(true);assert!(normalize(&body).is_err());
    }
    #[test]
    fn named_choice_keeps_prompt_tools_and_forces_one_call() {
        let body=json!({"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b","strict":true}}],"tool_choice":{"type":"function","function":{"name":"a"}},"parallel_tool_calls":true});
        let (tools,policy)=normalize(&body).unwrap();assert_eq!(tools.unwrap().as_array().unwrap().len(),2);let policy=policy.unwrap();assert!(policy.required&&policy.constrained&&!policy.parallel);assert_eq!(policy.schemas.len(),1);assert!(policy.strict.is_empty());
    }
    #[test]
    fn none_choice_validates_then_keeps_prompt_but_has_no_callable_schema() {
        let body=json!({"tools":[{"type":"function","function":{"name":"a","strict":true}}],"tool_choice":"none"});let (tools,policy)=normalize(&body).unwrap();assert!(tools.is_some());let policy=policy.unwrap();assert!(policy.schemas.is_empty()&&policy.constrained&&!policy.required);
    }
    #[test]
    fn names_duplicates_types_and_remote_references_are_refused() {
        for function in [json!({"name":" a"}),json!({"name":"工具"}),json!({"name":"a","strict":1}),json!({"name":"a","parameters":{"$ref":"https://example.invalid"}})] {assert!(normalize(&json!({"tools":[{"type":"function","function":function}]})).is_err());}
        assert!(normalize(&json!({"parallel_tool_calls":1})).is_err());
    }
}
