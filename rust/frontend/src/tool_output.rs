//! Incremental Qwen XML tool-call projection into OpenAI argument deltas.
use crate::{bad, tool_arguments, tools::{convert_value,schema_types,Policy},Result};
use serde_json::{json,Value};
use std::collections::{BTreeSet,HashMap};
const CALL:&str="<tool_call>\n<function=";
const CLOSE:&str="</tool_call>";
const FUNCTION:&str="</function>";
const PARAM:&str="<parameter=";
const PARAM_END:&str="\n</parameter>\n";
const THINK:&str="</think>";
type Types=Option<BTreeSet<String>>;
pub type Event=(&'static str,Value);
fn space(c:char)->bool {[' ','\t','\n','\r'].contains(&c)}
fn py_space(c:char)->bool {c.is_whitespace()||('\u{1c}'..='\u{1f}').contains(&c)}
fn encoded(value:&Value)->String {
    value.to_string().replace('\u{85}',"\\u0085").replace('\u{2028}',"\\u2028").replace('\u{2029}',"\\u2029")
}
fn held(text:&str,tags:&[&str])->usize {
    text.char_indices().map(|(i,_)|text.len()-i).filter(|n|tags.iter().any(|tag|*n<tag.len()&&tag.starts_with(&text[text.len()-n..]))).max().unwrap_or(0)
}
#[derive(Clone,Copy,PartialEq)]
enum State{Output,Json,Content,Name,Arguments,Parameter,Value,Closing}
pub struct Projector {
    request_id:String,pending:String,state:State,index:usize,call_id:Option<String>,name:Option<String>,
    schemas:HashMap<String,Value>,parameter_types:HashMap<String,Types>,other_types:Types,names:BTreeSet<String>,
    arguments:String,count:usize,parameter:Option<String>,types:Types,streams:bool,started:bool,parts:String,
    content:Vec<String>,streamed:usize,visible:bool,calls:Vec<Value>,
}
impl Projector {
    pub fn open_call_id(&self)->Option<String>{self.call_id.clone()}
    pub fn new(policy:&Policy,request_id:String,structured:bool)->Result<Self> {
        let mut schemas=HashMap::new();let mut budget=tool_arguments::BUDGET;
        for (name,schema) in &policy.schemas {
            let schema=if policy.strict.contains(name){tool_arguments::strict_schema(schema)}else{schema.clone()};
            match tool_arguments::frame(&schema,&mut budget) {
                Ok(schema)=>{schemas.insert(name.clone(),schema);},
                Err(error) if policy.strict.contains(name)=>return Err(error),_=>{}
            }
        }
        Ok(Self{request_id,pending:String::new(),state:if structured{State::Output}else{State::Content},index:0,call_id:None,name:None,schemas,parameter_types:HashMap::new(),other_types:None,names:BTreeSet::new(),arguments:String::new(),count:0,parameter:None,types:None,streams:false,started:false,parts:String::new(),content:Vec::new(),streamed:0,visible:false,calls:Vec::new()})
    }
    fn emit_content(&mut self,text:String,events:&mut Vec<Event>) {
        if text.is_empty(){return;}let visible=!text.trim_matches(py_space).is_empty();self.content.push(text);
        if !self.visible {if !visible{return;}self.visible=true;}
        let unsent=self.content[self.streamed..].concat();self.streamed=self.content.len();events.push(("content",json!(unsent)));
    }
    fn argument(&mut self,text:String,events:&mut Vec<Event>) {
        if self.call_id.is_none(){return;}self.arguments.push_str(&text);
        let mut start=0;let mut count=0;
        for (i,_) in text.char_indices(){if count==16384{events.push(("tool",json!({"index":self.index,"function":{"arguments":&text[start..i]}})));start=i;count=0;}count+=1;}
        if start<text.len(){events.push(("tool",json!({"index":self.index,"function":{"arguments":&text[start..]}})));}
    }
    fn begin_call(&mut self,name:String,closed:bool,events:&mut Vec<Event>) {
        if name.trim_matches(space).is_empty() {
            self.state=State::Content;
            self.emit_content(format!("{CALL}{name}{}",if closed {">"} else {""}),events);
            return;
        }
        self.state=State::Arguments;self.arguments.clear();self.count=0;self.names.clear();
        let name=name.trim_matches(space);
        if self.streamed==0{self.content.clear();}
        self.name=Some(name.to_string());self.call_id=Some(format!("call_{}_{}",self.request_id,self.index));
        self.parameter_types.clear();self.other_types=None;
        if let Some(schema)=self.schemas.get(name) {
            if let Some(properties)=schema["properties"].as_object(){for (key,value) in properties{self.parameter_types.insert(key.clone(),schema_types(value,value));}}
            let other=&schema["additionalProperties"];self.other_types=schema_types(other,other);
        }
        events.push(("tool",json!({"index":self.index,"id":self.call_id,"type":"function","function":{"name":name}})));
        self.argument("{".into(),events);
    }
    fn key(&mut self,name:&str)->String {
        let separator=if self.count==0{""}else{","};self.count+=1;format!("{separator}{}:",encoded(&json!(name)))
    }
    fn begin_parameter(&mut self,name:String,events:&mut Vec<Event>) {
        let name=name.trim_matches(space);let kept=!name.is_empty()&&!self.names.contains(name)&&self.call_id.is_some();self.names.insert(name.to_string());
        self.parameter=kept.then(||name.to_string());self.types=self.parameter_types.get(name).cloned().unwrap_or_else(||self.other_types.clone());
        self.streams=self.types==Some(BTreeSet::from(["string".into()]));self.started=false;self.parts.clear();
        if self.streams&&kept{let key=self.key(name);self.argument(key+"\"",events);}self.state=State::Value;
    }
    fn put_value(&mut self,text:&str,events:&mut Vec<Event>) {
        if self.parameter.is_none(){return;}
        let text=if !self.started {if text.is_empty(){return;}self.started=true;text.strip_prefix('\n').unwrap_or(text)}else{text};
        if text.is_empty(){return;}
        if self.streams {let encoded=encoded(&json!(text));self.argument(encoded[1..encoded.len()-1].to_string(),events);}else{self.parts.push_str(text);}
    }
    fn end_parameter(&mut self,text:&str,events:&mut Vec<Event>) {
        if let Some(name)=self.parameter.clone() {
            let text=if !self.started{self.started=true;text.strip_prefix('\n').unwrap_or(text)}else{text};self.put_value(text,events);
            if self.streams{self.argument("\"".into(),events);}else{let value=convert_value(&self.parts,self.types.as_ref());let key=self.key(&name);self.argument(key+&encoded(&value),events);}
        }
        self.parameter=None;self.parts.clear();self.state=State::Arguments;
    }
    fn finish_call(&mut self,events:&mut Vec<Event>) {
        self.argument("}".into(),events);
        if self.call_id.is_some(){self.calls.push(json!({"id":self.call_id,"type":"function","function":{"name":self.name,"arguments":self.arguments}}));self.index+=1;}
        self.call_id=None;self.name=None;self.arguments.clear();self.visible=false;self.state=State::Closing;
    }
    fn next_tag(&self,tags:&[&'static str])->Option<(usize,&'static str)> {tags.iter().filter_map(|tag|self.pending.find(tag).map(|i|(i,*tag))).min()}
    fn read_name(&mut self)->Option<String> {
        if let Some(end)=self.pending.find('>').filter(|i|self.pending[..*i].chars().count()<=256){let name=self.pending[..end].to_string();self.pending=self.pending[end+1..].to_string();Some(name)}
        else if self.pending.chars().count()<=256{None}else{Some(String::new())}
    }
    pub fn put(&mut self,text:&str)->Vec<Event> {
        self.pending.push_str(text);let mut events=Vec::new();
        while !self.pending.is_empty() {
            match self.state {
                State::Output=>{let first=self.pending.trim_start_matches(py_space);if first.is_empty(){break;}self.state=if first.starts_with('<'){State::Content}else{State::Json};},
                State::Json=>{let text=std::mem::take(&mut self.pending);self.emit_content(text,&mut events);break;},
                State::Content=>{
                    if let Some((start,tag))=self.next_tag(&[CALL,THINK]){let text=self.pending[..start].to_string();self.pending=self.pending[start+tag.len()..].to_string();self.emit_content(text,&mut events);if tag==CALL{self.state=State::Name;}}
                    else{let end=self.pending.len()-held(&self.pending,&[CALL,THINK]);let text=self.pending[..end].to_string();self.pending=self.pending[end..].to_string();self.emit_content(text,&mut events);break;}
                },
                State::Name=>{let closed=self.pending.find('>').is_some_and(|i|self.pending[..i].chars().count()<=256);let Some(name)=self.read_name()else{break;};self.begin_call(name,closed,&mut events);},
                State::Parameter=>{let Some(name)=self.read_name()else{break;};self.begin_parameter(name,&mut events);},
                State::Arguments=>{
                    if let Some((start,tag))=self.next_tag(&[PARAM,FUNCTION,CLOSE]){self.pending=self.pending[start+tag.len()..].to_string();if tag==PARAM{self.state=State::Parameter;}else{self.finish_call(&mut events);if tag==CLOSE{self.state=State::Content;}}}
                    else{let end=self.pending.len()-held(&self.pending,&[PARAM,FUNCTION,CLOSE]);self.pending=self.pending[end..].to_string();break;}
                },
                State::Value=>{
                    const ENDS:[&str;3]=["\n</parameter>\n<parameter=","\n</parameter>\n</function>","\n</parameter>\n</tool_call>"];
                    if let Some((start,_))=self.next_tag(&ENDS){let text=self.pending[..start].to_string();self.pending=self.pending[start+PARAM_END.len()..].to_string();self.end_parameter(&text,&mut events);}
                    else{let end=self.pending.len()-held(&self.pending,&ENDS);let text=self.pending[..end].to_string();self.pending=self.pending[end..].to_string();self.put_value(&text,&mut events);break;}
                },
                State::Closing=>{let rest=self.pending.trim_start_matches(space);if let Some(rest)=rest.strip_prefix(CLOSE){self.pending=rest.to_string();}else if CLOSE.starts_with(rest){break;}self.state=State::Content;},
            }
        }
        events
    }
    pub fn finish(&mut self,incomplete:bool)->(String,Vec<Value>,Vec<Event>) {
        let mut events=Vec::new();
        if !incomplete {
            if self.state==State::Name{let name=if self.pending.chars().count()<=256{std::mem::take(&mut self.pending)}else{self.pending.clear();String::new()};self.begin_call(name,false,&mut events);}
            if self.state==State::Value{self.pending.clear();self.end_parameter("",&mut events);}
            if [State::Arguments,State::Parameter].contains(&self.state){self.pending.clear();self.finish_call(&mut events);}
        }
        if self.state==State::Closing{self.pending.clear();self.state=State::Content;}
        if [State::Content,State::Output,State::Json].contains(&self.state)&&!self.pending.is_empty()&&!(incomplete&&[CALL,THINK].iter().any(|tag|tag.starts_with(&self.pending))){self.content.push(self.pending.clone());}
        else if !self.calls.is_empty(){self.content.truncate(self.streamed);}
        self.pending.clear();let mut calls=self.calls.clone();
        if self.call_id.is_some(){calls.push(json!({"id":self.call_id,"type":"function","function":{"name":self.name,"arguments":self.arguments}}));}
        let streamed=self.content[..self.streamed].concat();let content=if !calls.is_empty()&&incomplete{streamed.clone()}else{self.content.concat()};
        if content.len()>streamed.len(){events.push(("content",json!(&content[streamed.len()..])));}
        (content,calls,events)
    }
}
pub fn fixture(body:&Value)->Result<Value> {
    let (_,policy)=crate::tools::normalize(&body["request"])?;let policy=policy.ok_or_else(||bad("projector needs tool policy"))?;
    let mut projector=Projector::new(&policy,"fixture".into(),body["structured"]==true)?;
    let mut events=Vec::new();for part in body["parts"].as_array().ok_or_else(||bad("parts must be an array"))? {events.extend(projector.put(part.as_str().ok_or_else(||bad("part must be a string"))?));}
    let (content,calls,tail)=projector.finish(body["incomplete"]==true);events.extend(tail);Ok(json!({"content":content,"calls":calls,"events":events}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_split_preserves_large_integer_argument_and_tool_header() {
        let request=json!({"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"n":{"type":"integer"}},"additionalProperties":false}}}]});
        let text="<tool_call>\n<function=lookup>\n<parameter=n>\n18446744073709551617\n</parameter>\n</function>\n</tool_call>";
        for split in 0..=text.len(){let result=fixture(&json!({"request":request,"parts":[&text[..split],&text[split..]]})).unwrap();assert_eq!(result["content"],"");assert_eq!(result["calls"][0]["function"]["arguments"],"{\"n\":18446744073709551617}");}
    }
}

#[cfg(test)]
mod blank_name_regression {
    use super::*;
    #[test]
    fn blank_call_followed_by_valid_call_and_unclosed_name() {
        let request=json!({"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}]});
        let blank="<tool_call>\n<function=>raw</function>\n</tool_call>";
        let good="<tool_call>\n<function=lookup>\n<parameter=x>\n7\n</parameter>\n</function>\n</tool_call>";
        let text=format!("{blank}{good}");
        let parts=text.chars().map(|c|c.to_string()).collect::<Vec<_>>();
        let result=fixture(&json!({"request":request,"parts":parts})).unwrap();
        assert_eq!(result["content"],blank);
        assert_eq!(result["calls"].as_array().unwrap().len(),1);
        assert_eq!(result["calls"][0]["function"]["name"],"lookup");
        assert_eq!(result["calls"][0]["function"]["arguments"],"{\"x\":7}");
        let text="<tool_call>\n<function= ";
        let result=fixture(&json!({"request":request,"parts":[text]})).unwrap();
        assert_eq!(result["content"],text);
    }
    #[test]
    fn blank_names_preserve_text_at_every_utf8_split() {
        let request=json!({"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]});
        for name in ["", " ", "\t\r\n"] {
            let text=format!("前文<tool_call>\n<function={name}>\n<parameter=x>\n秘密\n</parameter>\n</function>\n</tool_call>尾文");
            for split in text.char_indices().map(|(i,_)|i).chain(std::iter::once(text.len())) {
                let result=fixture(&json!({"request":request,"parts":[&text[..split],&text[split..]]})).unwrap();
                assert_eq!(result["content"],text,"name={name:?}, split={split}");
                assert_eq!(result["calls"],json!([]));
            }
        }
    }
}
