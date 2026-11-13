//! Direct option-logit readout over the existing native score request protocol.
use super::{prepare_deadline, reply, Service};
use crate::{
    bad,
    engine::{u32_at, u64_at},
    generation::Prepared,
    judgments::{self, Encoded},
    ApiError, Result,
};
use axum::response::Response;
use axum::{http::StatusCode, response::IntoResponse};
use justfit_native_client_prototype::{DONE, PROGRESS, START};
use serde_json::{json, Value};
use std::{sync::Arc,time::{Duration, Instant}};
use tokio::sync::OwnedSemaphorePermit;
pub(super) struct Score {
    pub logits: Vec<f64>,
    pub prompt: u32,
    pub prefill_ms: f64,
    pub wall_ms: f64,
}
pub(super) fn options(body: &Value) -> Result<(Duration, u8)> {
    let timeout = crate::deadline::request(body)?;
    let priority = match body.get("priority").map_or(Some("normal"), Value::as_str) {
        Some("foreground") => 0,
        Some("normal") => 1,
        Some("background") => 2,
        _ => return Err(bad("priority must be foreground, normal, or background")),
    };
    Ok((timeout, priority))
}
pub(super) async fn execute(
    s: &Service,
    encoded: &Encoded,
    timeout: Duration,
    priority: u8,
    received: Instant,
) -> Result<Score> {
    let frame = justfit_native_client_prototype::score_request(
        s.engine.next_id(),
        &encoded.tokens,
        &encoded.slots,
    )
    .map_err(|_| bad("invalid score request"))?;
    let mut prepared = Prepared {
        received_at: None,
        frame,
        thinking: false,
        stop: Vec::new(),
        raw: false,
        budget: 0,
        timeout,
        public_id: String::new(),
        created_at: 0,
        vision_owner: None,
        constraint: None,
        tools: None,
        response_schema: None,
        internal_start: false,
        prompt_token_ids: None,
        rendered_prompt: None,
    };
    prepared.frame.payload[8] = priority;
    prepare_deadline(&mut prepared, received)?;
    let deadline = crate::deadline::tokio_deadline(prepared.timeout);
    let mut ticket = s.engine.submit(prepared.frame)?;
    let mut started = false;
    loop {
        let frame = crate::deadline::wait_until(deadline, ticket.next()).await??;
        match frame.kind {
            START if !started => {
                started = true;
            }
            PROGRESS => {}
            DONE if started => {
                let count = u32_at(&frame.payload, 41)? as usize;
                if frame.payload[8] != 0
                    || u32_at(&frame.payload, 13)? != 0
                    || count != encoded.slots.len()
                    || frame.payload.len() != 45 + count * 4
                {
                    return Err(ApiError(
                        502,
                        "invalid native score result".into(),
                        "protocol_error".into(),
                    ));
                }
                let logits = (0..count)
                    .map(|i| u32_at(&frame.payload, 45 + i * 4).map(|v| f32::from_bits(v) as f64))
                    .collect::<Result<Vec<_>>>()?;
                if logits.iter().any(|v| !v.is_finite()) {
                    return Err(ApiError(
                        502,
                        "non-finite native score logits".into(),
                        "engine_error".into(),
                    ));
                }
                return Ok(Score {
                    logits,
                    prompt: u32_at(&frame.payload, 9)?,
                    prefill_ms: u64_at(&frame.payload, 17)? as f64 / 1000.,
                    wall_ms: u64_at(&frame.payload, 33)? as f64 / 1000.,
                });
            }
            _ => {
                return Err(ApiError(
                    502,
                    "unexpected native score event".into(),
                    "protocol_error".into(),
                ))
            }
        }
    }
}
pub(super) async fn post(
    s: Service,
    body: Value,
    _permit: Arc<OwnedSemaphorePermit>,
    _body_permit: super::BodyPermit,
    received: Instant,
) -> Result<Response> {
    s.front.check_model(&body)?;
    let (timeout, priority) = options(&body)?;
    let input = body.clone();
    let front = s.front.clone();
    let preparation=s.preparation_queue.enter(timeout).await?;
    let encoded = tokio::task::spawn_blocking(move || {
        let _preparation=preparation;
        let messages = judgments::messages(&input)?;
        let labels = (0..input["options"].as_array().unwrap().len())
            .map(|i| ((b'A' + i as u8) as char).to_string())
            .collect::<Vec<_>>();
        judgments::encode(&front, messages, &labels)
    })
    .await
    .map_err(|_| bad("judgment preparation failed"))??;
    let result = execute(&s, &encoded, timeout, priority, received).await?;
    let probabilities = judgments::softmax(&result.logits)?;
    Ok(reply(
        json!({"id":body["id"],"option_ids":body["options"].as_array().unwrap().iter().map(|v|v["id"].clone()).collect::<Vec<_>>(),"probabilities":probabilities,"option_logits":result.logits,"input_tokens":result.prompt,"answer_token_ids":encoded.slots,"prompt_sha256":encoded.digest,"prompt_version":"direct-options-v1","model":{"id":s.front.response_model},"readout":judgments::READOUT,"probability_status":judgments::STATUS,"forward_seconds":result.prefill_ms/1000.,"total_seconds":result.wall_ms/1000.,"usage":{"prompt_tokens":result.prompt,"completion_tokens":0,"total_tokens":result.prompt}}),
    ))
}

pub(super) fn invalid_fields(errors: Vec<Value>) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        axum::Json(json!({"detail":errors})),
    )
        .into_response()
}
pub(super) async fn systemone(
    s: Service,
    body: Value,
    _permit: Arc<OwnedSemaphorePermit>,
    _body_permit: super::BodyPermit,
    received: Instant,
) -> Result<Response> {
    let mut validated = crate::systemone::validate(&body);
    let mut errors = Vec::new();
    if !body["model"].as_str().is_some_and(|v| !v.is_empty()) {
        errors.push(crate::systemone::error(
            &["model"],
            "field required",
            "missing",
        ));
    } else if !s.front.accepts_model(&body["model"]) {
        errors.push(crate::systemone::error(
            &["model"],
            &format!(
                "model {} is not served by this endpoint",
                body["model"].as_str().unwrap()
            ),
            "value_error",
        ));
    }
    errors.append(&mut validated.errors);
    let mut time_body = body.clone();
    time_body["priority"] = json!("normal");
    let (timeout, _) = options(&time_body)?;
    let priority = match options(&body) {
        Ok((_, priority)) => priority,
        Err(error) => {
            errors.push(crate::systemone::error(
                &["priority"],
                &error.1,
                "value_error",
            ));
            1
        }
    };
    if !errors.is_empty() {
        return Ok(invalid_fields(errors));
    }
    let front = s.front.clone();
    let preparation=s.preparation_queue.enter(timeout).await?;
    let prepared = tokio::task::spawn_blocking(move || -> Result<_> {
        let _preparation=preparation;
        let mut slots = None;
        let mut jobs = Vec::new();
        let mut total = 0;
        let mut errors = Vec::new();
        for (id, question) in validated.questions {
            if received.elapsed() >= timeout {
                return Err(ApiError(
                    504,
                    "request timed out".into(),
                    "request_timeout".into(),
                ));
            }
            if question.deterministic {
                jobs.push((id, question, None));
                continue;
            }
            if slots.is_none() {
                slots = Some(crate::systemone::slot_labels(&front)?);
            }
            let available = slots.as_ref().unwrap();
            if question.labels.len() > available.len() {
                errors.push(crate::systemone::error(
                    &["questions", &id, "criteria"],
                    &format!(
                        "the served tokenizer supports {} answer slots; {} were requested",
                        available.len(),
                        question.labels.len()
                    ),
                    "value_error",
                ));
                break;
            }
            let labels = &available[..question.labels.len()];
            let messages = crate::systemone::messages(&validated.state, &question, labels);
            let encoded = judgments::encode(&front, messages, labels)?;
            total += encoded.tokens.len();
            if total > 1 << 20 {
                errors.push(crate::systemone::error(
                    &["questions", &id],
                    "total prepared question tokens exceed 1048576",
                    "value_error",
                ));
                break;
            }
            jobs.push((id, question, Some(encoded)));
        }
        Ok((jobs, errors))
    })
    .await
    .map_err(|_| bad("question preparation failed"))??;
    let (jobs, errors) = prepared;
    if !errors.is_empty() {
        return Ok(invalid_fields(errors));
    }
    let mut answers = serde_json::Map::new();
    let mut tokens = 0u64;
    for (id, question, encoded) in jobs {
        let probabilities = if let Some(encoded) = encoded {
            let result = execute(&s, &encoded, timeout, priority, received).await?;
            tokens += result.prompt as u64;
            judgments::softmax(&result.logits)?
        } else {
            vec![1.]
        };
        answers.insert(id, crate::systemone::answer(&question, &probabilities));
    }
    Ok(reply(
        json!({"model":s.front.response_model,"answers":answers,"usage":{"input_tokens":tokens,"output_tokens":0}}),
    ))
}
