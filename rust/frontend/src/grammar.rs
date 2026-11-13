//! Request-owned LLGuidance constraints, with bounded lazy compilation.
use crate::{bad, ApiError, Result};
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory};
use serde_json::{json, Value};
use std::{collections::VecDeque, path::PathBuf, sync::{Arc, Mutex}};
use tokio::sync::Semaphore;
use toktrie_hf_tokenizers::ByteTokenizer;

const WORDS: usize = 7760;
const SOURCE_BUDGET: usize = 8 * 1024 * 1024;
// LLGuidance expands a multipleOf into a state per residue (999983 compiles in
// seconds into hundreds of MiB), so larger divisors are checked on the
// complete output only. Array bounds are counted repetitions that compile in
// proportion to their values (10^5 items: ~20 ms, ~30 MiB; 10^7: seconds, over
// a GiB): the grammar keeps them up to a total per schema, and a maxItems past
// the longest output, which cannot bind, leaves it.
const MAX_GRAMMAR_DIVISOR: f64 = 64.;
const MAX_GRAMMAR_ITEMS: f64 = 262_144.;
const GRAMMAR_ITEMS_BUDGET: f64 = 262_144.;

// The bounds of `schema` the grammar takes (the validator checks the rest on
// the complete output): array bounds up to GRAMMAR_ITEMS_BUDGET in all, the
// smaller ones first, and divisors up to MAX_GRAMMAR_DIVISOR.
fn grammar_bounds(schema: &mut Value) {
    // Visits every object in document order; `edit` sees each array bound
    // with its index among them and says whether it stays.
    fn visit(value: &mut Value, next: &mut usize, edit: &mut impl FnMut(usize, f64) -> bool) {
        match value {
            Value::Object(map) => {
                if map.get("multipleOf").and_then(Value::as_f64).is_some_and(|v| v > MAX_GRAMMAR_DIVISOR) { map.remove("multipleOf"); }
                for key in ["minItems", "maxItems"] {
                    if let Some(v) = map.get(key).and_then(Value::as_f64) {
                        let index = *next; *next += 1;
                        if !edit(index, v) { map.remove(key); }
                    }
                }
                for child in map.values_mut() { visit(child, next, edit); }
            }
            Value::Array(array) => for child in array { visit(child, next, edit); },
            _ => {}
        }
    }
    let mut bounds = Vec::new();
    visit(schema, &mut 0, &mut |index, v| { bounds.push((v, index)); true });
    // The smaller bounds first; equal ones in document order, so equal schemas compile equally.
    bounds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let mut keep = vec![false; bounds.len()];
    let mut total = 0.;
    for &(v, index) in &bounds {
        if v <= 64. { keep[index] = true; continue; }
        if v > MAX_GRAMMAR_ITEMS || total + v > GRAMMAR_ITEMS_BUDGET { continue; }
        total += v; keep[index] = true;
    }
    visit(schema, &mut 0, &mut |index, _| keep[index]);
}
pub type Shared = Arc<Mutex<Session>>;
struct Template { key: String, matcher: Matcher }
struct Compiler { factory: Option<ParserFactory>, cache: VecDeque<Template>, bytes: usize, hits:u64, misses:u64 }
impl Compiler {
    fn get(&mut self,key:&str,validator:Arc<jsonschema::Validator>)->Option<Shared>{
        let index=self.cache.iter().position(|template|template.key==key)?;
        let template=self.cache.remove(index).unwrap();
        let session=Session{matcher:template.matcher.deep_clone(),validator,last_mask:0};
        self.cache.push_back(template);self.hits+=1;Some(Arc::new(Mutex::new(session)))
    }
    fn keep(&mut self,key:String,matcher:Matcher,validator:Arc<jsonschema::Validator>)->Shared{
        let session=Session{matcher:matcher.deep_clone(),validator:validator.clone(),last_mask:0};self.misses+=1;
        if key.len()<=SOURCE_BUDGET{
            while self.cache.len()>=32||self.bytes+key.len()>SOURCE_BUDGET{if let Some(old)=self.cache.pop_front(){self.bytes-=old.key.len();}else{break}}
            self.bytes+=key.len();self.cache.push_back(Template{key,matcher});
        }
        Arc::new(Mutex::new(session))
    }
}
pub struct Factory { tokenizer: PathBuf, compiler: Mutex<Compiler>, slots: Arc<Semaphore> }
pub struct Session { matcher: Matcher, pub validator: Arc<jsonschema::Validator>, last_mask: u64 }

pub(crate) fn schema_children(value: &mut Value, visit: &mut impl FnMut(&mut Value) -> Result<()>) -> Result<()> {
    visit(value)?;
    let Some(object) = value.as_object_mut() else { return Ok(()); };
    for (key, value) in object {
        if ["properties","patternProperties","$defs","definitions","dependentSchemas","dependencies"].contains(&key.as_str()) {
            if let Some(map) = value.as_object_mut() {
                for child in map.values_mut() {
                    if key != "dependencies" || !child.is_array() { schema_children(child, visit)?; }
                }
            }
        } else if ["items","prefixItems","additionalItems","contains","additionalProperties","unevaluatedItems","unevaluatedProperties","propertyNames","allOf","anyOf","oneOf","not","if","then","else","contentSchema","extends"].contains(&key.as_str()) {
            if let Some(array) = value.as_array_mut() {
                for child in array { schema_children(child,visit)?; }
            } else { schema_children(value,visit)?; }
        } else if ["type","disallow"].contains(&key.as_str()) {
            if let Some(array) = value.as_array_mut() {
                for child in array { if child.is_object() { schema_children(child,visit)?; } }
            }
        }
    }
    Ok(())
}
pub fn response_schema(body: &Value) -> Result<Option<Value>> {
    let value=&body["response_format"];
    if value.is_null() || value==&json!({"type":"text"}) { return Ok(None); }
    if !value.is_object() { return Err(bad("response_format must be an object")); }
    let mut schema=match value["type"].as_str() {
        Some("json_object") => json!({"type":"object"}),
        Some("json_schema") if value["json_schema"]["schema"].is_object() || value["json_schema"]["schema"].is_boolean() => value["json_schema"]["schema"].clone(),
        Some("json_schema") => return Err(bad("response_format.json_schema.schema is required")),
        _ => return Err(bad("unsupported response_format")),
    };
    let mut pending=vec![(&schema,1)];
    while let Some((value,depth))=pending.pop() {
        if (value.is_object() || value.is_array()) && depth>64 { return Err(bad("invalid response schema: nested more than 64 levels deep")); }
        match value { Value::Object(map)=>pending.extend(map.values().map(|v|(v,depth+1))), Value::Array(array)=>pending.extend(array.iter().map(|v|(v,depth+1))), _=>{} }
    }
    if schema.to_string().len()>SOURCE_BUDGET { return Err(bad("output schema exceeds compilation source budget")); }
    let dialect=|name:&str|name.trim_end_matches('#').replace("http://","https://");
    let root_dialect=dialect(schema.get("$schema").and_then(Value::as_str).unwrap_or("https://json-schema.org/draft/2020-12/schema"));
    let mut objects=vec![&schema];let mut patterns=false;let mut unevaluated=false;
    while let Some(value)=objects.pop() {
        match value {
            Value::Object(map)=>{patterns|=map.contains_key("patternProperties");unevaluated|=map.contains_key("unevaluatedProperties");objects.extend(map.values());},
            Value::Array(array)=>objects.extend(array),_=>{}
        }
    }
    if patterns && unevaluated {return Err(bad("unevaluatedProperties with patternProperties is not supported"));}
    schema_children(&mut schema,&mut |value| {
        if let Some(object)=value.as_object() {
            if object.get("$schema").is_some_and(|v|!v.is_string()) { return Err(bad("$schema must be a string")); }
            if object.get("$schema").and_then(Value::as_str).is_some_and(|name|dialect(name)!=root_dialect) {return Err(bad("mixed schema dialects are not supported"));}
            for key in ["$ref","$dynamicRef","$recursiveRef"] {
                if let Some(value)=object.get(key) {
                    let reference=value.as_str().ok_or_else(||bad(format!("{key} must be a string")))?;
                    if !reference.starts_with('#') { return Err(bad(format!("remote schema reference is not allowed: {reference}"))); }
                }
            }
        }
        Ok(())
    })?;
    Ok(Some(schema))
}

impl Factory {
    pub fn new(tokenizer: PathBuf) -> Self { Self { tokenizer,compiler:Mutex::new(Compiler{factory:None,cache:VecDeque::new(),bytes:0,hits:0,misses:0}),slots:Arc::new(Semaphore::new(4)) } }
    pub fn snapshot(&self)->Value{let compiler=self.compiler.lock().unwrap();json!({"entries":compiler.cache.len(),"capacity":32,"source_bytes":compiler.bytes,"source_budget_bytes":SOURCE_BUDGET,"hits":compiler.hits,"misses":compiler.misses})}
    fn initialize(&self,compiler:&mut Compiler)->Result<()>{
        if compiler.factory.is_none() {
            let mut tokenizer=ByteTokenizer::from_file(&self.tokenizer).map_err(|_|bad("grammar tokenizer could not be loaded"))?;
            tokenizer.set_eos_tokens(&[248044,248046]);let env=tokenizer.into_tok_env(Some(248320)).map_err(|_|bad("invalid grammar tokenizer"))?;
            let mut factory=ParserFactory::new_simple(&env).map_err(|_|bad("grammar compiler initialization failed"))?;factory.quiet();compiler.factory=Some(factory);
        }Ok(())
    }
    pub async fn create(self: &Arc<Self>, schema: Value, thinking: bool) -> Result<Shared> {
        let permit=self.slots.clone().acquire_owned().await.map_err(|_|bad("grammar compiler closed"))?;
        let this=self.clone();
        tokio::task::spawn_blocking(move || { let _permit=permit;this.compile(schema,thinking) }).await.map_err(|_|bad("grammar compiler failed"))?
    }
    pub async fn create_tools(self:&Arc<Self>,policy:Arc<crate::tools::Policy>,schema:Option<Value>,thinking:bool)->Result<Shared> {
        let permit=self.slots.clone().acquire_owned().await.map_err(|_|bad("grammar compiler closed"))?;
        let this=self.clone();tokio::task::spawn_blocking(move||{
            let _permit=permit;
            let mut compiler=this.compiler.lock().map_err(|_|bad("grammar compiler unavailable"))?;
            this.initialize(&mut compiler)?;
            let factory=compiler.factory.as_ref().unwrap();
            let mut compatible=|mut schema:Value|->Result<Value>{
                schema_children(&mut schema,&mut |value| {
                    if let Some(map)=value.as_object_mut(){map.remove("propertyNames");if let Some(pattern)=map.get("pattern").and_then(Value::as_str){let test=TopLevelGrammar::from_lark(format!("%llguidance {{}}\nstart: %json {}\n",json!({"type":"string","pattern":pattern})));if factory.create_parser(test).is_err(){map.remove("pattern");}}}Ok(())
                })?;
                grammar_bounds(&mut schema);
                if schema==true{schema=json!({});}
                if let Some(map)=schema.as_object_mut(){map.insert("x-guidance".into(),json!({"lenient":true,"whitespace_pattern":"[ \\t\\n\\r]{0,64}"}));}Ok(schema)
            };
            let grammar=crate::tool_grammar::build(&policy,thinking,schema.as_ref(),&mut compatible)?;
            if grammar.to_string().len()>16*1024*1024{return Err(bad("tool parameter schemas are too complex"));}
            let key=grammar.to_string();
            let validator=Arc::new(jsonschema::validator_for(&json!(true)).map_err(|_|bad("validator initialization failed"))?);
            if let Some(session)=compiler.get(&key,validator.clone()){return Ok(session)}
            let grammar:TopLevelGrammar=serde_json::from_value(grammar).map_err(|_|bad("invalid tool grammar"))?;
            let matcher=Matcher::new(compiler.factory.as_ref().unwrap().create_parser(grammar));
            if matcher.is_error(){return Err(bad(format!("unsupported tool schema: {}",matcher.get_error().unwrap_or_default().lines().next().unwrap_or("compiler failed"))));}
            Ok(compiler.keep(key,matcher,validator))
        }).await.map_err(|_|bad("tool grammar compiler failed"))?
    }
    fn compile(&self, schema: Value, thinking: bool) -> Result<Shared> {
        let mut compiler=self.compiler.lock().map_err(|_|bad("grammar compiler unavailable"))?;
        self.initialize(&mut compiler)?;
        let factory=compiler.factory.as_ref().unwrap();let mut compatible=schema.clone();
        schema_children(&mut compatible,&mut |value| {
            if let Some(object)=value.as_object_mut() {
                object.remove("propertyNames");
                if let Some(pattern)=object.get("pattern").and_then(Value::as_str) {
                    let test=TopLevelGrammar::from_lark(format!("%llguidance {{}}\nstart: %json {}\n",json!({"type":"string","pattern":pattern})));
                    if factory.create_parser(test).is_err() { object.remove("pattern"); }
                }
            }
            Ok(())
        })?;
        grammar_bounds(&mut compatible);if compatible==true {compatible=json!({});}
        if let Some(object)=compatible.as_object_mut(){object.insert("x-guidance".into(),json!({"lenient":true,"whitespace_pattern":"[ \\t\\n\\r]{0,64}"}));}
        let thinking_rules=if thinking {"think: TEXT <[248069]>\nTEXT: /(?s:.*)/ & ~/(?s:.*)<\\/think>(?s:.*)/\n"} else {""};
        let grammar=format!("%llguidance {{}}\nstart: {}WS %json {}\n{}WS: /[ \\t\\n\\r]{{0,64}}/\n",if thinking{"think "}else{""},compatible,thinking_rules);
        // Full validation belongs to this request's original schema. Multiple
        // schemas can share the same lenient compiled grammar and token masks.
        // Network/file resolution is compiled out of this dependency.
        let validator=Arc::new(jsonschema::validator_for(&schema).map_err(|error|bad(format!("invalid response schema: {error}")))?);
        if let Some(session)=compiler.get(&grammar,validator.clone()){return Ok(session)}
        let matcher=Matcher::new(compiler.factory.as_ref().unwrap().create_parser(TopLevelGrammar::from_lark(grammar.clone())));
        if matcher.is_error() { return Err(bad(format!("unsupported output schema: {}",matcher.get_error().unwrap_or_default().lines().next().unwrap_or("compiler failed")))); }
        Ok(compiler.keep(grammar,matcher,validator))
    }
}
impl Session {
    #[cfg(test)]
    fn clone_for_test(&self) -> Session {
        Session { matcher: self.matcher.deep_clone(), validator: self.validator.clone(), last_mask: self.last_mask }
    }
    pub fn commit(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.iter().any(|t|*t>=248320) { return Err(bad("generated token is out of range")); }
        let stopped_eos=tokens.len()==1 && [248044,248046].contains(&tokens[0]) && !self.matcher.is_error() && self.matcher.is_stopped() && self.matcher.is_accepting().unwrap_or(false);
        if !stopped_eos {self.matcher.consume_tokens(tokens).map_err(|_|ApiError(502,"output grammar rejected generated tokens".into(),"constraint_error".into()))?;}
        Ok(())
    }
    /// masks() for a draft tree: row 0 is the anchor and every row's parent
    /// row is smaller. The rows are the state before the anchor and after
    /// each row's path; a row its parent's grammar refuses takes its parent's
    /// mask (the target's choice after the parent cannot be that row).
    pub fn masks_tree(&mut self, request_id:u64, mask_id:u64, tokens:&[u32], parents:&[u32]) -> Result<Vec<u8>> {
        if mask_id==0 || mask_id<=self.last_mask || tokens.is_empty() || tokens.len()>16 || parents.len()!=tokens.len()
            || parents[0]!=0 || parents.iter().enumerate().skip(1).any(|(row,parent)|*parent as usize>=row) {
            return Err(bad("invalid tree mask request"));
        }
        self.last_mask=mask_id;
        let mask=|state:&mut Matcher|->Result<Vec<u32>>{
            let mask=state.compute_mask_or_eos().map_err(|_|bad("output grammar has no valid token"))?;let words=mask.as_slice();
            if words.len()<WORDS || words[WORDS..].iter().any(|w|*w!=0) || !words[..WORDS].iter().any(|w|*w!=0) {return Err(bad("invalid vocabulary mask"));}
            Ok(words[..WORDS].to_vec())
        };
        let mut before=self.matcher.deep_clone();
        let mut rows:Vec<Vec<u32>>=vec![mask(&mut before)?];
        // Each row's state after its path, or None past a refused token.
        let mut states:Vec<Option<Matcher>>=Vec::with_capacity(tokens.len());
        for (row,&token) in tokens.iter().enumerate() {
            let parent=if row==0 {Some(self.matcher.deep_clone())} else {states[parents[row] as usize].as_ref().map(|s|s.deep_clone())};
            let state=parent.and_then(|mut state|{
                let stopped_eos=[248044,248046].contains(&token)&&!state.is_error()&&state.is_stopped()&&state.is_accepting().unwrap_or(false);
                if stopped_eos {return Some(state);}
                (token<248320 && state.validate_tokens(&[token]).ok()==Some(1) && state.consume_token(token).is_ok()).then_some(state)
            });
            let words=match state.as_ref() {
                Some(_)=>mask(&mut state.as_ref().unwrap().deep_clone())?,
                None=>rows[if row==0 {0} else {parents[row] as usize+1}].clone(),
            };
            rows.push(words);states.push(state);
        }
        let words:Vec<u32>=rows.concat();
        let mut bytes=request_id.to_le_bytes().to_vec();bytes.extend(mask_id.to_le_bytes());bytes.extend((words.len() as u32).to_le_bytes());for word in words{bytes.extend(word.to_le_bytes());}Ok(bytes)
    }
    pub fn masks(&mut self, request_id:u64, mask_id:u64, simulation:&[u32]) -> Result<Vec<u8>> {
        if mask_id==0 || mask_id<=self.last_mask || simulation.len()>=9 {return Err(bad("invalid mask request"));}self.last_mask=mask_id;
        let in_range=simulation.iter().position(|t|*t>=248320).unwrap_or(simulation.len());
        let valid=self.matcher.validate_tokens(&simulation[..in_range]).map_err(|_|bad("invalid speculative mask state"))?;
        let mut trial=self.matcher.deep_clone();let mut rows=Vec::with_capacity((simulation.len()+1)*WORDS);let mut last=Vec::new();
        for step in 0..=valid {
            if step>0 {let token=simulation[step-1];let stopped_eos=[248044,248046].contains(&token)&&!trial.is_error()&&trial.is_stopped()&&trial.is_accepting().unwrap_or(false);if !stopped_eos {trial.consume_token(token).map_err(|_|bad("invalid speculative token"))?;}}
            let mask=trial.compute_mask_or_eos().map_err(|_|bad("output grammar has no valid token"))?;let words=mask.as_slice();
            if words.len()<WORDS || words[WORDS..].iter().any(|w|*w!=0) || !words[..WORDS].iter().any(|w|*w!=0) {return Err(bad("invalid vocabulary mask"));}
            last=words[..WORDS].to_vec();rows.extend_from_slice(&last);
        }
        for _ in valid..simulation.len(){rows.extend_from_slice(&last);}
        let mut bytes=request_id.to_le_bytes().to_vec();bytes.extend(mask_id.to_le_bytes());bytes.extend((rows.len() as u32).to_le_bytes());for word in rows{bytes.extend(word.to_le_bytes());}Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn references_in_schema_are_refused_but_literal_data_is_not_a_schema() {
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"$ref":"file:///secret"}}}})).is_err());
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"const":{"$ref":"literal"}}}}})).is_ok());
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"properties":{"child":{"$dynamicRef":"https://example.invalid"}}}}}})).is_err());
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":false}}})).is_ok());
    }

    fn digits(items: u64) -> Value {
        json!({"type":"array","items":{"type":"integer","minimum":0,"maximum":9},"minItems":items,"maxItems":items})
    }

    #[test]
    fn array_bounds_stay_in_the_grammar_within_the_budget() {
        // The paper singleton's long_digits schema keeps its exact length.
        let mut schema = digits(10_000);
        grammar_bounds(&mut schema);
        assert_eq!(schema, digits(10_000));
        // A maxItems past the longest output cannot bind and leaves the grammar.
        let mut schema = json!({"type":"array","minItems":3,"maxItems":10_000_000});
        grammar_bounds(&mut schema);
        assert_eq!(schema, json!({"type":"array","minItems":3}));
        // Past the per-schema total, the larger bounds go to the validator alone.
        let mut schema = json!({"type":"object","properties":{
            "a":{"type":"array","minItems":200_000},"b":{"type":"array","minItems":40_000},
            "c":{"type":"array","minItems":40_000},"d":{"type":"array","maxItems":64}}});
        grammar_bounds(&mut schema);
        assert_eq!(schema["properties"]["a"], json!({"type":"array"}));
        assert_eq!(schema["properties"]["b"], json!({"type":"array","minItems":40_000}));
        assert_eq!(schema["properties"]["c"], json!({"type":"array","minItems":40_000}));
        assert_eq!(schema["properties"]["d"], json!({"type":"array","maxItems":64}));
        // Of equal bounds the earlier in the document stays.
        let mut schema = json!({"type":"array","items":{"type":"array","minItems":140_000},"minItems":140_000});
        grammar_bounds(&mut schema);
        assert_eq!(schema, json!({"type":"array","items":{"type":"array"},"minItems":140_000}));
        // Divisors keep their previous bound.
        let mut schema = json!({"anyOf":[{"type":"integer","multipleOf":97},{"type":"number","multipleOf":7.5}]});
        grammar_bounds(&mut schema);
        assert_eq!(schema, json!({"anyOf":[{"type":"integer"},{"type":"number","multipleOf":7.5}]}));
    }

    // With the model's tokenizer (JUSTFIT_TEST_TOKENIZER, a directory holding
    // tokenizer.json): the grammar refuses to close a long_digits array early,
    // which the validator rejected after generation as invalid_model_output.
    #[tokio::test]
    async fn long_array_cannot_close_before_min_items() {
        let Some(directory) = std::env::var_os("JUSTFIT_TEST_TOKENIZER") else {
            eprintln!("skipped: JUSTFIT_TEST_TOKENIZER is not set");
            return;
        };
        let path = std::path::PathBuf::from(directory).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
        let ids = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let factory = Arc::new(Factory::new(path.clone()));
        for items in [100u64, 10_000] {
            let session = factory.create(digits(items), false).await.unwrap();
            let mut session = session.lock().unwrap();
            session.commit(&ids("[1")).unwrap();
            assert!(session.clone_for_test().commit(&ids("]")).is_err(), "{items} items: closed after one");
            let rest: String = (1..items).map(|i| format!(", {}", i % 10)).collect();
            session.commit(&ids(&rest)).unwrap();
            assert!(session.clone_for_test().commit(&ids(", 1")).is_err(), "{items} items: grew past maxItems");
            session.commit(&ids("]")).unwrap();
        }
    }

    // masks_tree against masks(): a chain-shaped tree's payload is the
    // chain's, and each branch of a tree takes the masks of its path as a
    // chain, a refused branch its parent's.
    #[tokio::test]
    async fn tree_masks_are_the_masks_of_their_paths() {
        let Some(directory) = std::env::var_os("JUSTFIT_TEST_TOKENIZER") else {
            eprintln!("skipped: JUSTFIT_TEST_TOKENIZER is not set");
            return;
        };
        let path = std::path::PathBuf::from(directory).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
        let ids = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let schema = json!({"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","items":{"enum":["a","b"]},"maxItems":3}},"required":["name","tags"],"additionalProperties":false});
        let factory = Arc::new(Factory::new(path.clone()));
        let session = factory.create(schema, false).await.unwrap();
        let mut session = session.lock().unwrap();
        session.commit(&ids("{\"name\": \"Ada\"")).unwrap();
        // The anchor and a chain of five tokens.
        let chain: Vec<u32> = ids(", \"tags\": [\"a\"");
        assert!(chain.len() >= 4, "{chain:?}");
        let payload = |bytes: Vec<u8>| bytes[16..].to_vec();  // past request id and mask id
        let mut a = session.clone_for_test();
        let mut b = session.clone_for_test();
        let parents: Vec<u32> = (0..chain.len() as u32).map(|r| r.saturating_sub(1)).collect();
        assert_eq!(payload(a.masks(7, 1, &chain).unwrap()), payload(b.masks_tree(7, 1, &chain, &parents).unwrap()));
        // A tree: the chain, a refused sibling of row 2 and a valid sibling of row 1.
        let refused = ids(" }")[0];
        let mut tokens = chain.clone();
        let mut tree_parents = parents.clone();
        tokens.push(refused); tree_parents.push(1);
        let other = ids(", \"tags\": []");
        tokens.push(other[1]); tree_parents.push(0);
        let mut c = session.clone_for_test();
        let tree = payload(c.masks_tree(7, 1, &tokens, &tree_parents).unwrap());
        let rows = |bytes: &[u8]| -> Vec<Vec<u8>> { bytes[4..].chunks(WORDS * 4).map(|r| r.to_vec()).collect() };
        let tree_rows = rows(&tree);
        assert_eq!(tree_rows.len(), tokens.len() + 1);
        // The refused row (its parent is row 1) takes row 1's mask.
        assert_eq!(tree_rows[chain.len() + 1], tree_rows[2]);
        // The sibling of row 1 takes the masks of the chain [anchor, sibling].
        let mut d = session.clone_for_test();
        let sibling_chain = rows(&payload(d.masks(7, 1, &[chain[0], other[1]]).unwrap()));
        assert_eq!(tree_rows[chain.len() + 2], sibling_chain[2]);
    }
}
