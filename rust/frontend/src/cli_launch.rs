//! `justfit launch claude|codex`: start a coding agent wired to a running
//! JustFit server. Nothing is written to the agent's own configuration: Claude
//! Code gets environment variables, Codex gets `-c` overrides, so the user's
//! normal setup is untouched once the agent exits.
use serde_json::Value;
use std::collections::BTreeMap;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const HELP: &str = "usage: justfit launch claude|codex [--url URL] [--model NAME] [--dry-run] [-- AGENT_ARGS...]\n\n  claude          Claude Code through the Anthropic Messages API\n  codex           Codex CLI through the OpenAI Responses API\n  --url URL       Running JustFit server; default http://127.0.0.1:$JUSTFIT_PORT or :8000\n  --model NAME    Served model name; default the server's first model\n  --dry-run       Print the command and environment without starting the agent\n\nStart the server first with justfit serve. JUSTFIT_API_KEY, if set, is passed\nto the agent. The agent's own configuration files are not modified.";

pub struct Plan {
    pub program: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// TOML basic string (Codex parses `-c key=value` values as TOML).
fn toml_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn plan(agent: &str, url: &str, model: &str, context: Option<u64>, key: &str, extra: &[String]) -> Result<Plan> {
    let url = url.trim_end_matches('/');
    let mut env = BTreeMap::new();
    let (program, mut args) = match agent {
        "claude" => {
            env.insert("ANTHROPIC_BASE_URL".into(), url.to_string());
            env.insert("ANTHROPIC_AUTH_TOKEN".into(), key.to_string());
            env.insert("ANTHROPIC_MODEL".into(), model.to_string());
            env.insert("ANTHROPIC_DEFAULT_OPUS_MODEL".into(), model.to_string());
            env.insert("ANTHROPIC_DEFAULT_SONNET_MODEL".into(), model.to_string());
            env.insert("ANTHROPIC_DEFAULT_HAIKU_MODEL".into(), model.to_string());
            env.insert("ANTHROPIC_SMALL_FAST_MODEL".into(), model.to_string());
            ("claude".to_string(), Vec::new())
        }
        "codex" => {
            env.insert("JUSTFIT_API_KEY".into(), key.to_string());
            let mut args = Vec::new();
            let mut set = |k: &str, v: String| {
                args.push("-c".to_string());
                args.push(format!("{k}={v}"));
            };
            set("model_provider", toml_string("justfit"));
            set("model_providers.justfit.name", toml_string("JustFit (local)"));
            set("model_providers.justfit.base_url", toml_string(&format!("{url}/v1")));
            set("model_providers.justfit.wire_api", toml_string("responses"));
            set("model_providers.justfit.env_key", toml_string("JUSTFIT_API_KEY"));
            set("model", toml_string(model));
            // A local server cannot run OpenAI's hosted web search; JustFit
            // would drop the tool anyway, so do not offer it to the model.
            set("web_search", toml_string("disabled"));
            if let Some(context) = context {
                set("model_context_window", context.to_string());
            }
            ("codex".to_string(), args)
        }
        other => return Err(format!("unknown agent {other:?}: use claude or codex").into()),
    };
    args.extend(extra.iter().cloned());
    Ok(Plan { program, args, env })
}

fn discover(url: &str, key: &str) -> Result<(String, Option<u64>)> {
    let models = format!("{}/v1/models", url.trim_end_matches('/'));
    let mut request = ureq::get(&models);
    if !key.is_empty() && key != "justfit" {
        request = request.header("authorization", &format!("Bearer {key}"));
    }
    let mut response = request.call().map_err(|e| {
        format!("no JustFit server answered at {url} ({e}); start one with: justfit serve")
    })?;
    let body: Value = serde_json::from_reader(response.body_mut().as_reader())?;
    let first = &body["data"][0];
    let name = first["id"].as_str().ok_or("the server listed no models")?.to_string();
    Ok((name, first["context_length"].as_u64()))
}

pub fn run(args: &[String]) -> Result<()> {
    let mut agent = None;
    let mut url = None;
    let mut model = None;
    let mut dry = false;
    let mut extra = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        match arg {
            "--" => {
                extra.extend(args[index + 1..].iter().cloned());
                break;
            }
            "--dry-run" => dry = true,
            "--url" | "--model" => {
                let value = args.get(index + 1).ok_or(format!("{arg} needs a value"))?.clone();
                if arg == "--url" { url = Some(value) } else { model = Some(value) }
                index += 1;
            }
            _ if agent.is_none() && !arg.starts_with('-') => agent = Some(arg.to_string()),
            _ => return Err(format!("unexpected argument {arg}; run justfit launch --help").into()),
        }
        index += 1;
    }
    let agent = agent.ok_or("usage: justfit launch claude|codex; run justfit launch --help")?;
    let url = url.unwrap_or_else(|| {
        format!("http://127.0.0.1:{}", std::env::var("JUSTFIT_PORT").unwrap_or_else(|_| "8000".into()))
    });
    let key = std::env::var("JUSTFIT_API_KEY").ok().filter(|k| !k.is_empty()).unwrap_or_else(|| "justfit".into());
    let (served, context) = discover(&url, &key)?;
    let model = model.unwrap_or(served);
    let plan = plan(&agent, &url, &model, context, &key, &extra)?;
    if dry {
        for (name, value) in &plan.env {
            let shown = if name.contains("TOKEN") || name.contains("KEY") { "[redacted]" } else { value.as_str() };
            println!("{name}={shown}");
        }
        println!("{} {}", plan.program, plan.args.join(" "));
        return Ok(());
    }
    use std::os::unix::process::CommandExt;
    let error = std::process::Command::new(&plan.program).args(&plan.args).envs(&plan.env).exec();
    if error.kind() == std::io::ErrorKind::NotFound {
        let hint = if agent == "claude" {
            "install Claude Code: curl -fsSL https://claude.ai/install.sh | bash"
        } else {
            "install Codex: brew install --cask codex (or npm install -g @openai/codex)"
        };
        return Err(format!("{} is not on PATH; {hint}", plan.program).into());
    }
    Err(error.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_plan_uses_overrides_not_config_files() {
        let p = plan("codex", "http://127.0.0.1:8000/", "m/odel", Some(262_144), "k", &["exec".into(), "hi".into()]).unwrap();
        assert_eq!(p.program, "codex");
        let joined = p.args.join(" ");
        assert!(joined.contains("model_providers.justfit.base_url=\"http://127.0.0.1:8000/v1\""), "{joined}");
        assert!(joined.contains("model_providers.justfit.wire_api=\"responses\""));
        assert!(joined.contains("web_search=\"disabled\""));
        assert!(joined.contains("model_context_window=262144"));
        assert!(joined.contains("model=\"m/odel\""));
        assert_eq!(&p.args[p.args.len() - 2..], ["exec", "hi"]);
        assert_eq!(p.env["JUSTFIT_API_KEY"], "k");
    }

    #[test]
    fn claude_plan_points_every_model_slot_at_the_server() {
        let p = plan("claude", "http://h:9", "m", None, "k", &[]).unwrap();
        assert_eq!(p.program, "claude");
        assert_eq!(p.env["ANTHROPIC_BASE_URL"], "http://h:9");
        for slot in ["ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"] {
            assert_eq!(p.env[slot], "m");
        }
        assert!(p.args.is_empty());
    }

    #[test]
    fn toml_strings_escape_quotes() {
        assert_eq!(toml_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
    }

    #[test]
    fn unknown_agent_is_rejected() {
        assert!(plan("cursor", "http://h", "m", None, "k", &[]).is_err());
    }
}
