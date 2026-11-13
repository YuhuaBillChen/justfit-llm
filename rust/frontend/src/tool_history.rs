//! Preserve a valid truncated JSON call as literal history, as in 2.3.
use serde_json::{json,Value};
#[derive(Clone,Copy,PartialEq,Eq)]
enum Next{Value,Done,ArrayFirst,ArrayValue,ArrayEnd,ObjectFirst,Key,Colon,ObjectValue,ObjectEnd}
pub fn unfinished(text:&str)->bool{
    use Next::*;
    let bytes=text.as_bytes();let size=bytes.len();let mut pos=0;let mut stack=vec![Value];
    while pos<size{
        let byte=bytes[pos];if b" \t\r\n".contains(&byte){pos+=1;continue}
        let expected=*stack.last().expect("root completion sentinel");
        if expected==ArrayEnd||expected==ObjectEnd{
            if byte==if expected==ArrayEnd{b']'}else{b'}'}{stack.pop();}
            else if byte==b','{*stack.last_mut().unwrap()=if expected==ArrayEnd{ArrayValue}else{Key};}
            else{return false}pos+=1;continue
        }
        if expected==Colon{if byte!=b':'{return false}*stack.last_mut().unwrap()=ObjectValue;pos+=1;continue}
        if (expected==ArrayFirst&&byte==b']')||(expected==ObjectFirst&&byte==b'}'){stack.pop();pos+=1;continue}
        if expected==Done{return false}
        let key=expected==Key||expected==ObjectFirst;if key&&byte!=b'"'{return false}
        *stack.last_mut().unwrap()=if key{Colon}else if expected==Value{Done}else if expected==ObjectValue{ObjectEnd}else{ArrayEnd};
        match byte{
            b'['|b'{'=>{stack.push(if byte==b'['{ArrayFirst}else{ObjectFirst});pos+=1;},
            b'"'=>{
                pos+=1;
                loop{
                    if pos==size{return true}let byte=bytes[pos];pos+=1;
                    if byte==b'"'{break}
                    if byte<32{return false}
                    if byte==b'\\'{
                        if pos==size{return true}let escape=bytes[pos];pos+=1;
                        if escape==b'u'{let end=(pos+4).min(size);if !bytes[pos..end].iter().all(u8::is_ascii_hexdigit){return false}
                            if end-pos<4{return true}pos=end;}
                        else if !b"\"\\/bfnrt".contains(&escape){return false}
                    }
                }
            },
            b't'|b'f'|b'n'=>{
                let literal:&[u8]=match byte{b't'=>b"true",b'f'=>b"false",_=>b"null"};let length=literal.len().min(size-pos);
                if literal[..length]!=bytes[pos..pos+length]{return false}pos+=length;if length<literal.len(){return true}
            },
            _=>{
                let start=pos;if bytes[pos]==b'-'{pos+=1;}
                if pos==size||!bytes[pos].is_ascii_digit(){return byte==b'-'&&start+1==size}
                if bytes[pos]==b'0'{pos+=1;}else{while pos<size&&bytes[pos].is_ascii_digit(){pos+=1;}}
                if pos+1<size&&bytes[pos]==b'.'&&bytes[pos+1].is_ascii_digit(){pos+=2;while pos<size&&bytes[pos].is_ascii_digit(){pos+=1;}}
                if pos<size&&matches!(bytes[pos],b'e'|b'E'){
                    let mut end=pos+1;if end<size&&matches!(bytes[end],b'+'|b'-'){end+=1;}
                    if end<size&&bytes[end].is_ascii_digit(){pos=end+1;while pos<size&&bytes[pos].is_ascii_digit(){pos+=1;}}
                }
                if size-pos<=2{
                    let suffix=&bytes[pos..];let exponent=bytes[start..pos].iter().any(|b|matches!(b,b'e'|b'E'));
                    if [b"e".as_slice(),b"E",b"e+",b"e-",b"E+",b"E-"].contains(&suffix){return !exponent}
                    if suffix==b"."{return !exponent&&!bytes[start..pos].contains(&b'.')}
                }
            },
        }
    }
    stack!=[Next::Done]
}
fn has_content(message:&Value)->bool{
    ["content","reasoning_content","tool_calls"].iter().any(|key|message[*key].as_str().is_some_and(|s|!s.is_empty())||message[*key].as_array().is_some_and(|a|!a.is_empty()))
}
pub fn template_messages(messages:Vec<Value>)->Value{
    let mut output=Vec::new();
    for message in messages{
        let Some(calls)=message["tool_calls"].as_array().filter(|calls|calls.iter().any(|call|call["function"]["arguments"].is_string()))else{output.push(message);continue};
        let mut current=message.clone();current.as_object_mut().unwrap().remove("tool_calls");
        for call in calls{
            if !call["function"]["arguments"].is_string(){current.as_object_mut().unwrap().entry("tool_calls").or_insert(json!([])).as_array_mut().unwrap().push(call.clone());continue}
            if has_content(&current){output.push(current)}
            output.push(json!({"role":"assistant","content":format!("Incomplete tool call:\n{}",crate::python_json::dumps(&call["function"]))}));
            current=json!({"role":"assistant","content":""});
        }
        if has_content(&current){output.push(current)}
    }
    json!(output)
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn distinguishes_prefixes_from_malformed_or_complete_json_without_recursion(){
        for text in ["", "{", "[1,", "{\"x\":tru", "{\"x\":1e+", "\"\\u12"]{assert!(unfinished(text),"{text}")}
        for text in ["{}","[]","{x", "[1,]", "{\"x\":1e+}", "true false", "\"\\q", "01"]{assert!(!unfinished(text),"{text}")}
        assert!(unfinished(&"[".repeat(100000)));
    }
}
