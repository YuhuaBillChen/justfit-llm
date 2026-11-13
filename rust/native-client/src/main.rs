use justfit_native_client_prototype::{read_frame, text_request, write_frame, Client};
use std::{env, fs, io};

fn tokens(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(fs::read_to_string(path)?.split_whitespace().map(str::parse).collect::<Result<Vec<_>, _>>()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("roundtrip") if args.len() == 1 => {
            let mut input = io::stdin().lock(); let mut output = io::stdout().lock();
            while let Some(frame) = read_frame(&mut input)? { write_frame(&mut output, &frame)?; }
        }
        Some("encode-text") if args.len() == 5 => {
            let frame = text_request(1, &tokens(&args[1])?, args[2].parse()?, args[3].parse()?, args[4].parse()?)?;
            write_frame(&mut io::stdout().lock(), &frame)?;
        }
        Some("run") if args.len() >= 5 && args[3] == "--" => {
            let prompt = tokens(&args[1])?; let long = tokens(&args[2])?;
            let mut client = Client::spawn(&args[4..])?;
            let first = client.run(1, &prompt, 64, None)?;
            let repeat = client.run(2, &prompt, 64, None)?;
            let cancelled = client.run(3, &long, 2048, Some(1))?;
            let followup = client.run(4, &prompt, 64, None)?;
            if first.tokens != repeat.tokens || first.tokens != followup.tokens || cancelled.reason != 2 {
                return Err("repeat/cancel/follow-up mismatch".into());
            }
            println!("{{\"ready\":{{\"lanes\":{},\"context\":{},\"vision\":{}}},\"tokens\":{:?},\"repeat_tokens\":{:?},\"followup_tokens\":{:?},\"reason\":{},\"prompt_tokens\":{},\"cancel_reason\":{},\"cancel_tokens\":{}}}",
                client.lanes, client.context, client.vision, first.tokens, repeat.tokens, followup.tokens,
                first.reason, first.prompt, cancelled.reason, cancelled.tokens.len());
        }
        _ => return Err("usage: roundtrip | encode-text TOKENS MAX THINK END | run TOKENS CANCEL_TOKENS -- NATIVE_COMMAND...".into()),
    }
    Ok(())
}
