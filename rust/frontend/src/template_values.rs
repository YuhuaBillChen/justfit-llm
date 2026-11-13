//! Lossless bridge from arbitrary-precision JSON into MiniJinja values.
use minijinja::{value::{Kwargs,Object,ObjectRepr,Value,ValueKind},Error,ErrorKind};
use serde::Serialize;
use serde_json::Value as Json;
use std::{fmt,sync::Arc};
#[derive(Debug)]
struct BigInteger(String);
impl Object for BigInteger {
    fn repr(self:&Arc<Self>)->ObjectRepr{ObjectRepr::Plain}
    fn render(self:&Arc<Self>,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str(&self.0)}
}
pub fn value(input:&Json)->Value {
    match input {
        Json::Null=>Value::from(()),Json::Bool(v)=>Value::from(*v),Json::String(v)=>Value::from(v.clone()),
        Json::Array(v)=>Value::from_iter(v.iter().map(value)),
        Json::Object(v)=>Value::make_object_map(v.clone(),|map|Box::new(map.keys().map(|key|Value::from(key.clone()))),|map,key|map.get(key.as_str()?).map(value)),
        Json::Number(v)=>{
            let text=v.to_string();
            if text.contains(['.','e','E']){Value::from(v.as_f64().expect("finite JSON number"))}
            else if let Ok(v)=text.parse::<i128>(){Value::from(v)}else if let Ok(v)=text.parse::<u128>(){Value::from(v)}else{Value::from_object(BigInteger(text))}
        }
    }
}
fn has_big(value:&Value)->bool {
    if value.downcast_object_ref::<BigInteger>().is_some(){return true;}
    match value.kind(){ValueKind::Map=>value.try_iter().is_ok_and(|mut keys|keys.any(|k|value.get_item(&k).is_ok_and(|v|has_big(&v)))),ValueKind::Seq=>value.try_iter().is_ok_and(|mut items|items.any(|v|has_big(&v))),_=>false}
}
fn json_value(value:&Value)->Result<Json,Error> {
    let error=||Error::new(ErrorKind::InvalidOperation,"cannot serialize template value to JSON");
    if let Some(big)=value.downcast_object_ref::<BigInteger>(){return serde_json::from_str(&big.0).map_err(|_|error());}
    match value.kind() {
        ValueKind::Map=>{let mut map=serde_json::Map::new();for key in value.try_iter()?{map.insert(key.to_string(),json_value(&value.get_item(&key)?)?);}Ok(Json::Object(map))},
        ValueKind::Seq=>Ok(Json::Array(value.try_iter()?.map(|v|json_value(&v)).collect::<Result<_,_>>()?)),
        _=>serde_json::to_value(value).map_err(|_|error()),
    }
}
struct Spaced;
impl serde_json::ser::Formatter for Spaced {
    fn begin_array_value<W:?Sized+std::io::Write>(&mut self,writer:&mut W,first:bool)->std::io::Result<()>{if first{Ok(())}else{writer.write_all(b", ")}}
    fn begin_object_key<W:?Sized+std::io::Write>(&mut self,writer:&mut W,first:bool)->std::io::Result<()>{if first{Ok(())}else{writer.write_all(b", ")}}
    fn begin_object_value<W:?Sized+std::io::Write>(&mut self,writer:&mut W)->std::io::Result<()>{writer.write_all(b": ")}
}
pub fn tojson(value:&Value,indent:Option<Value>,kwargs:Kwargs)->Result<Value,Error> {
    if !has_big(value){return minijinja::filters::tojson(value,indent,kwargs);}
    let indent=match indent{Some(v)=>Some(v),None=>kwargs.get("indent")?};kwargs.assert_all_used()?;
    let indent=match indent {None=>None,Some(v)=>match bool::try_from(v.clone()).ok(){Some(true)=>Some(2),Some(false)=>None,None=>Some(usize::try_from(v)?)} };
    let input=json_value(value)?;let mut bytes=Vec::new();
    if let Some(indent)=indent {
        if indent>1024{return Err(Error::new(ErrorKind::InvalidOperation,"JSON indentation is too large"));}
        let spaces=vec![b' ';indent];let formatter=serde_json::ser::PrettyFormatter::with_indent(&spaces);input.serialize(&mut serde_json::Serializer::with_formatter(&mut bytes,formatter))
    }else{input.serialize(&mut serde_json::Serializer::with_formatter(&mut bytes,Spaced))}.map_err(|_|Error::new(ErrorKind::InvalidOperation,"cannot serialize JSON"))?;
    let text=String::from_utf8(bytes).map_err(|_|Error::new(ErrorKind::InvalidOperation,"invalid JSON encoding"))?;
    Ok(Value::from_safe_string(text.replace('<',"\\u003c").replace('>',"\\u003e").replace('&',"\\u0026").replace('\'',"\\u0027")))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn template_json_preserves_small_large_and_beyond_128_bit_numbers() {
        for number in ["1","18446744073709551617","100000000000000000000000000000000000000000000000001"] {
            let input=crate::strict_json(format!("{{\"n\":{number},\"s\":\"<hello>\"}}").as_bytes()).unwrap();
            let mut env=minijinja::Environment::new();env.add_filter("tojson",tojson);
            let text=env.render_str("{{ v|tojson }}",minijinja::context!(v=>value(&input))).unwrap();assert_eq!(text,format!("{{\"n\": {number}, \"s\": \"\\u003chello\\u003e\"}}"));
        }
    }
}
