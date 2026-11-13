//! SystemOne validation and conditional-option readout semantics.
use crate::{judgments, Frontend, Result};
use serde_json::{json, Value};
#[derive(Clone)]
pub struct Question {
    pub kind: String,
    pub instructions: Option<Value>,
    pub labels: Vec<String>,
    pub descriptions: Vec<Value>,
    pub legend: Value,
    pub deterministic: bool,
}
pub struct Validated {
    pub state: Value,
    pub questions: Vec<(String, Question)>,
    pub errors: Vec<Value>,
}
fn field(loc: Vec<Value>, message: &str, kind: &str) -> Value {
    let mut path = vec![json!("body")];
    path.extend(loc);
    json!({"loc":path,"msg":message,"type":kind})
}
pub fn error(path: &[&str], message: &str, kind: &str) -> Value {
    field(path.iter().map(|s| json!(s)).collect(), message, kind)
}
fn description(value: &Value) -> bool {
    value.is_null() || value.is_string() || value.is_object() || value.is_array()
}
fn question(id: &str, value: &Value) -> (Option<Question>, Vec<Value>) {
    let mut errors = Vec::new();
    let loc = vec![json!("questions"), json!(id)];
    if !value.is_object() {
        return (
            None,
            vec![field(loc, "question must be an object", "model_type")],
        );
    }
    let Some(kind) = value["type"]
        .as_str()
        .filter(|k| ["noul", "choice", "score"].contains(k))
    else {
        let mut path = loc;
        path.push(json!("type"));
        return (
            None,
            vec![field(
                path,
                "type must be noul, choice, or score",
                "value_error",
            )],
        );
    };
    let instructions = value.get("instructions").filter(|v| !v.is_null()).cloned();
    if instructions
        .as_ref()
        .is_some_and(|v| !v.is_string() && !v.is_object() && !v.is_array())
    {
        let mut path = loc.clone();
        path.push(json!("instructions"));
        errors.push(field(
            path,
            "instructions must be a string, object, or array",
            "value_error",
        ));
    }
    let criteria = &value["criteria"];
    let mut labels = Vec::new();
    let mut descriptions = Vec::new();
    let mut legend = Value::Null;
    let mut deterministic = false;
    let mut issue = None;
    if kind == "noul" {
        if !criteria.is_null()
            && (!criteria.is_object()
                || ["true", "false"]
                    .iter()
                    .any(|key| !description(&criteria[key])))
        {
            issue = Some("noul criteria must map true/false to a string, object, array, or null");
        }
        labels = vec!["true".into(), "false".into()];
        descriptions = labels
            .iter()
            .map(|key| {
                criteria
                    .get(key)
                    .filter(|v| !v.is_null())
                    .cloned()
                    .unwrap_or(json!(key))
            })
            .collect();
    } else if kind == "choice" {
        match criteria.as_object() {
            None => {
                issue =
                    Some("choice criteria must be a nonempty object mapping labels to descriptions")
            }
            Some(map) if map.is_empty() => {
                issue =
                    Some("choice criteria must be a nonempty object mapping labels to descriptions")
            }
            Some(map) if map.len() > 255 => issue = Some("choice supports at most 255 options"),
            Some(map) if map.values().any(|v| !description(v)) => {
                issue = Some("choice descriptions must be strings, objects, arrays, or null")
            }
            Some(map) => {
                labels = map.keys().cloned().collect();
                descriptions = map
                    .iter()
                    .map(|(k, v)| if v.is_null() { json!(k) } else { v.clone() })
                    .collect();
                deterministic = labels.len() == 1;
            }
        }
    } else {
        match criteria.as_array() {
            None => issue = Some("score criteria must be a nonempty array of level descriptions"),
            Some(array) if array.is_empty() => {
                issue = Some("score criteria must be a nonempty array of level descriptions")
            }
            Some(array) if array.len() > 255 => issue = Some("score supports at most 255 levels"),
            Some(array)
                if array
                    .iter()
                    .any(|v| !v.is_string() && !v.is_array() && !v.is_object()) =>
            {
                issue = Some("score descriptions must be strings, objects, or arrays")
            }
            Some(array) => {
                labels = (0..array.len()).map(|i| i.to_string()).collect();
                descriptions = array.clone();
                legend = json!(labels
                    .iter()
                    .cloned()
                    .zip(descriptions.iter().cloned())
                    .collect::<serde_json::Map<_, _>>());
                deterministic = array.len() == 1;
            }
        }
    }
    if let Some(issue) = issue {
        let mut path = loc;
        path.push(json!("criteria"));
        errors.push(field(path, issue, "value_error"));
    }
    if !errors.is_empty() {
        return (None, errors);
    }
    (
        Some(Question {
            kind: kind.into(),
            instructions,
            labels,
            descriptions,
            legend,
            deterministic,
        }),
        errors,
    )
}
pub fn validate(body: &Value) -> Validated {
    let mut errors = Vec::new();
    let mut state = body.get("state").cloned().unwrap_or(Value::Null);
    if body.get("state").is_none() {
        errors.push(error(&["state"], "field required", "missing"));
    } else if !state.is_string() && !state.is_array() && !state.is_object() {
        errors.push(error(
            &["state"],
            "state must be a string, object, or array",
            "value_error",
        ));
        state = Value::Null;
    }
    let mut questions = Vec::new();
    match body.get("questions") {
        None => errors.push(error(&["questions"], "field required", "missing")),
        Some(value) => match value.as_object() {
            None => errors.push(error(
                &["questions"],
                "questions must be a nonempty object",
                "value_error",
            )),
            Some(map) if map.is_empty() => errors.push(error(
                &["questions"],
                "questions must be a nonempty object",
                "value_error",
            )),
            Some(map) if map.len() > 64 => errors.push(error(
                &["questions"],
                "questions must contain at most 64 entries",
                "value_error",
            )),
            Some(map) => {
                for (id, value) in map {
                    let (question, issues) = question(id, value);
                    errors.extend(issues);
                    if let Some(question) = question {
                        questions.push((id.clone(), question));
                    }
                }
            }
        },
    }
    Validated {
        state,
        questions,
        errors,
    }
}
pub fn messages(state: &Value, question: &Question, slots: &[String]) -> Value {
    let mut payload = json!({"evidence":state});
    if let Some(instructions) = &question.instructions {
        payload["criterion"] = instructions.clone();
    }
    payload["options"]=json!(slots.iter().zip(question.labels.iter()).zip(question.descriptions.iter()).map(|((slot,label),description)|json!({"slot":slot,"label":label,"description":description})).collect::<Vec<_>>());
    json!([{"role":"system","content":judgments::DIRECT.replace("uppercase letter","uppercase slot")},{"role":"user","content":crate::python_json::dumps(&payload)}])
}
pub fn answer(question: &Question, probabilities: &[f64]) -> Value {
    let concentration = if probabilities.len() < 2 {
        1.
    } else {
        let entropy = -judgments::compensated(
            probabilities
                .iter()
                .filter(|p| **p > 0.)
                .map(|p| p * p.ln()),
        );
        (1. - entropy / (probabilities.len() as f64).ln()).max(0.)
    };
    let map = question
        .labels
        .iter()
        .cloned()
        .zip(probabilities.iter().map(|v| json!(v)))
        .collect::<serde_json::Map<_, _>>();
    if question.kind == "noul" {
        json!({"type":"noul","noul":probabilities[0]})
    } else if question.kind == "choice" {
        let best = probabilities
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .map(|(i, _)| i)
            .unwrap();
        json!({"type":"choice","choice":question.labels[best],"probabilities":map,"confidence":concentration})
    } else {
        json!({"type":"score","score":judgments::compensated(probabilities.iter().enumerate().map(|(i,p)|i as f64*p)),"legend":question.legend,"probabilities":map,"confidence":concentration})
    }
}
pub fn fixture(body: &Value) -> Value {
    let validated = validate(body);
    let questions=validated.questions.iter().map(|(id,q)|json!({"id":id,"kind":q.kind,"instructions":q.instructions,"labels":q.labels,"descriptions":q.descriptions,"legend":q.legend,"deterministic":q.deterministic})).collect::<Vec<_>>();
    json!({"state":validated.state,"questions":questions,"errors":validated.errors})
}
pub fn slot_labels(front: &Frontend) -> Result<Vec<String>> {
    front.score_labels.get_or_init(||derive_slots(front)).clone()
}
fn derive_slots(front:&Frontend)->Result<Vec<String>> {
    let mut labels = Vec::new();
    for length in 1..=3 {
        for index in 0..26usize.pow(length) {
            let mut n = index;
            let mut bytes = vec![b'A'; length as usize];
            for byte in bytes.iter_mut().rev() {
                *byte += (n % 26) as u8;
                n /= 26;
            }
            let label = String::from_utf8(bytes).unwrap();
            let tokens = front.encode(&label, false)?;
            if tokens.len() == 1
                && front.tokenizer.decode(&tokens, false).ok().as_deref() == Some(label.as_str())
            {
                labels.push(label);
                if labels.len() == 255 {
                    return Ok(labels);
                }
            }
        }
    }
    Ok(labels)
}
pub fn prepare_fixture(front:&Frontend,body:&Value)->Result<Value> {
    let validated=validate(body);if !validated.errors.is_empty(){return Ok(json!({"errors":validated.errors}));}
    let available=slot_labels(front)?;let mut jobs=Vec::new();
    for (id,question) in validated.questions {
        if question.deterministic{jobs.push(json!({"id":id,"deterministic":true,"answer":answer(&question,&[1.])}));continue;}
        if question.labels.len()>available.len(){return Err(crate::bad("too many answer slots"));}
        let labels=&available[..question.labels.len()];let messages=messages(&validated.state,&question,labels);let encoded=judgments::encode(front,messages.clone(),labels)?;
        let frame=justfit_native_client_prototype::score_request(1,&encoded.tokens,&encoded.slots).map_err(|_|crate::bad("invalid score frame"))?;let mut bytes=Vec::new();justfit_native_client_prototype::write_frame(&mut bytes,&frame).map_err(|_|crate::bad("invalid score frame"))?;
        jobs.push(json!({"id":id,"deterministic":false,"messages":messages,"prompt":encoded.prompt,"tokens":encoded.tokens,"slots":encoded.slots,"wire":bytes}));
    }
    Ok(json!({"jobs":jobs}))
}
