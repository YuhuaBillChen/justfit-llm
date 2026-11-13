//! Bounded projection of JSON object schemas into Qwen XML parameters.
use crate::{bad, grammar::schema_children, tools::{children,lookup}, Result};
use serde_json::{json,Map,Value};
use std::collections::BTreeSet;

pub const BUDGET:usize=16*1024*1024;
pub fn charge(budget:&mut usize,size:usize)->Result<()> {
    *budget=budget.checked_sub(size).ok_or_else(||bad("tool parameter schemas are too complex"))?;Ok(())
}
fn size(value:&Value,budget:usize)->usize {
    let mut pending=vec![value];let mut size=0;
    while size<=budget {
        let Some(value)=pending.pop() else{break;};
        match value {
            Value::String(text)=>size+=text.len(),
            Value::Array(array)=>{pending.extend(array);size+=1;},
            Value::Object(map)=>{size+=1+map.keys().map(String::len).sum::<usize>();pending.extend(map.values());},
            _=>size+=1,
        }
    }
    size
}
fn schemas(root:&Value)->Vec<&Value> {
    let mut pending=vec![root];let mut out=Vec::new();
    while let Some(value)=pending.pop(){out.push(value);pending.extend(children(value));}out
}
pub fn strict_schema(root:&Value)->Value {
    let annotations=["$comment","title","description","default","examples","deprecated","readOnly","writeOnly","$schema","$id","$anchor","$dynamicAnchor","$defs","definitions"];
    for value in schemas(root) {
        if let Some(map)=value.as_object() {
            let keys=map.keys().map(String::as_str).filter(|k|!annotations.contains(k)).collect::<BTreeSet<_>>();
            let applicators=["allOf","not","if","then","else","dependentSchemas","dependencies","extends","$dynamicRef","$recursiveRef"];
            if applicators.iter().any(|k|keys.contains(k)) ||(keys.contains("$ref")&&keys.len()!=1) ||(keys.iter().any(|k|["anyOf","oneOf"].contains(k))&&keys.iter().any(|k|!["anyOf","oneOf","type"].contains(k))){return root.clone();}
        }
    }
    let mut output=root.clone();
    schema_children(&mut output,&mut |value| {
        if let Some(map)=value.as_object_mut() {
            if ["anyOf","oneOf","$ref"].iter().any(|k|map.contains_key(*k)){return Ok(());}
            let has_kind=|kind:&str|map.get("type").is_some_and(|v|v==kind||v.as_array().is_some_and(|a|a.iter().any(|v|v==kind)));
            let object=has_kind("object")||(map.get("type").is_none_or(Value::is_null)&&map.contains_key("properties"));
            let array=has_kind("array")||(map.get("type").is_none_or(Value::is_null)&&map.contains_key("prefixItems"));
            if object && !["additionalProperties","unevaluatedProperties","patternProperties"].iter().any(|k|map.contains_key(*k)){map.insert("additionalProperties".into(),json!(false));}
            if array {
                if map.get("items").is_some_and(Value::is_array) {
                    if !["additionalItems","unevaluatedItems"].iter().any(|k|map.contains_key(*k)){map.insert("additionalItems".into(),json!(false));}
                } else if !["items","unevaluatedItems"].iter().any(|k|map.contains_key(*k)){map.insert("items".into(),json!(false));}
            }
        }Ok(())
    }).expect("infallible schema traversal");output
}
pub fn with_root(schema:&Value,root:&Value)->Value {
    if !schemas(schema).iter().any(|v|v.get("$ref").and_then(Value::as_str).is_some_and(|r|r.starts_with('#'))){return schema.clone();}
    let mut name="__justfit_root".to_string();
    while root.get("$defs").and_then(|v|v.get(&name)).is_some()||schema.get("$defs").and_then(|v|v.get(&name)).is_some(){name.push('_');}
    let prefix=format!("#/$defs/{name}");
    let rebase=|value:&Value| {
        let mut output=value.clone();schema_children(&mut output,&mut |value| {
            if let Some(reference)=value.get("$ref").and_then(Value::as_str) {
                if reference=="#"||reference.starts_with("#/"){value["$ref"]=json!(format!("{prefix}{}",&reference[1..]));}
            }Ok(())
        }).expect("infallible rebasing");output
    };
    let mut output=rebase(schema);let definitions=output.as_object_mut().unwrap().entry("$defs").or_insert(json!({}));definitions[&name]=rebase(root);output
}
fn combination(keyword:&str,values:Vec<Value>)->Value {
    let identity=keyword=="allOf";let mut kept=Vec::new();
    for value in values {
        if value==identity{continue;}
        if value.is_boolean(){return json!(!identity);}
        kept.push(value);
    }
    if kept.is_empty(){json!(identity)}else if kept.len()==1{kept.pop().unwrap()}else{json!({keyword:kept})}
}
#[derive(Clone)]
struct Shape {properties:Map<String,Value>,required:BTreeSet<String>,additional:Value}
fn combine(shapes:Vec<Option<Shape>>,union:bool,budget:&mut usize)->Result<Option<Shape>> {
    if !union&&shapes.iter().any(Option::is_none){return Ok(None);}
    let shapes=shapes.into_iter().flatten().collect::<Vec<_>>();
    if union&&shapes.is_empty(){return Ok(None);}
    if shapes.is_empty(){return Ok(Some(Shape{properties:Map::new(),required:BTreeSet::new(),additional:json!(true)}));}
    let mut names=Vec::new();for shape in &shapes {for name in shape.properties.keys(){if !names.contains(name){names.push(name.clone());}}}
    charge(budget,names.len().saturating_mul(shapes.len()))?;
    let mut required=shapes[0].required.clone();
    for shape in &shapes[1..]{if union{required=required.intersection(&shape.required).cloned().collect();}else{required.extend(shape.required.iter().cloned());}}
    let keyword=if union{"anyOf"}else{"allOf"};let mut properties=Map::new();
    for name in names {properties.insert(name.clone(),combination(keyword,shapes.iter().map(|s|s.properties.get(&name).unwrap_or(&s.additional).clone()).collect()));}
    Ok(Some(Shape{properties,required,additional:combination(keyword,shapes.into_iter().map(|s|s.additional).collect())}))
}
fn object_keyword<'a>(node:&'a Value,key:&str)->Result<Option<&'a Map<String,Value>>> {
    node.get(key).map(|v|v.as_object().ok_or_else(||bad(format!("unsupported tool parameter schema: {key}")))).transpose()
}
fn array_keyword<'a>(node:&'a Value,key:&str)->Result<Option<&'a Vec<Value>>> {
    node.get(key).map(|v|v.as_array().ok_or_else(||bad(format!("unsupported tool parameter schema: {key}")))).transpose()
}
fn project(node:&Value,root:&Value,visiting:&BTreeSet<String>,budget:&mut usize,depth:usize)->Result<Option<Shape>> {
    if depth>128{return Err(bad("tool parameter schemas are too complex"));}
    if node==false{return Ok(None);}
    let empty=json!({});let node=if node==true{&empty}else{node};
    if !node.is_object(){return Err(bad("tool parameters must allow a JSON object"));}
    if node.get("type").is_some_and(|kind|kind!="object"&&!kind.as_array().is_some_and(|a|a.iter().any(|v|v=="object"))){return Ok(None);}
    let mut properties=object_keyword(node,"properties")?.cloned().unwrap_or_default();
    let mut additional=node.get("additionalProperties").cloned().unwrap_or(json!(true));
    if let Some(patterns)=object_keyword(node,"patternProperties")?.filter(|m|!m.is_empty()){additional=combination("anyOf",std::iter::once(additional).chain(patterns.values().cloned()).collect());}
    let required=array_keyword(node,"required")?.cloned().unwrap_or_default();let mut names=BTreeSet::new();
    for name in required {let name=name.as_str().ok_or_else(||bad("unsupported tool parameter schema: required"))?;names.insert(name.to_string());properties.entry(name).or_insert(additional.clone());}
    let mut shapes=vec![Some(Shape{properties,required:names,additional})];
    if let Some(reference)=node.get("$ref") {
        let reference=reference.as_str().ok_or_else(||bad("unsupported tool parameter reference"))?;
        if visiting.contains(reference){return Err(bad("cyclic direct tool argument reference"));}
        let mut next=visiting.clone();next.insert(reference.to_string());shapes.push(project(lookup(reference,root)?,root,&next,budget,depth+1)?);
    }
    for child in array_keyword(node,"allOf")?.into_iter().flatten(){shapes.push(project(child,root,visiting,budget,depth+1)?);}
    for keyword in ["anyOf","oneOf"] {
        if let Some(members)=array_keyword(node,keyword)? {
            let children=members.iter().map(|v|project(v,root,visiting,budget,depth+1)).collect::<Result<Vec<_>>>()?;
            shapes.push(combine(children,true,budget)?);
        }
    }
    if node.get("if").is_some(){
        let a=project(node.get("then").unwrap_or(&empty),root,visiting,budget,depth+1)?;
        let b=project(node.get("else").unwrap_or(&empty),root,visiting,budget,depth+1)?;
        shapes.push(combine(vec![a,b],true,budget)?);
    }
    for child in object_keyword(node,"dependentSchemas")?.into_iter().flat_map(|m|m.values()) {
        let a=project(child,root,visiting,budget,depth+1)?;let b=project(&empty,root,visiting,budget,depth+1)?;shapes.push(combine(vec![a,b],true,budget)?);
    }
    let enumeration=node.get("const").map(|v|vec![v]).or_else(||node.get("enum").and_then(Value::as_array).map(|a|a.iter().collect()));
    if let Some(values)=enumeration {
        let choices=values.into_iter().filter_map(Value::as_object).map(|map|Some(Shape{properties:map.iter().map(|(k,v)|(k.clone(),json!({"const":v}))).collect(),required:map.keys().cloned().collect(),additional:json!(false)})).collect::<Vec<_>>();
        if choices.is_empty(){return Ok(None);}shapes.push(combine(choices,true,budget)?);
    }
    combine(shapes,false,budget)
}
pub fn frame(root:&Value,budget:&mut usize)->Result<Value> {
    let shape=project(root,root,&BTreeSet::new(),budget,0)?.ok_or_else(||bad("tool parameters must allow a top-level JSON object"))?;
    let mut properties=Map::new();
    let mut framed=|value:&Value|->Result<Value>{charge(budget,size(value,*budget))?;let result=with_root(value,root);if &result!=value{charge(budget,size(root,*budget))?;}Ok(result)};
    for (name,value) in shape.properties{properties.insert(name,framed(&value)?);}
    let additional=framed(&shape.additional)?;
    Ok(json!({"properties":properties,"required":shape.required,"additionalProperties":additional,"type":"object"}))
}
pub fn fixture(body:&Value)->Result<Value> {
    let root=if body["strict"]==true{strict_schema(&body["schema"])}else{body["schema"].clone()};let mut budget=BUDGET;frame(&root,&mut budget)
}
