use justfit_rust_frontend::{json_body, Frontend};
use serde_json::json;
use std::{
    io::{self, BufRead, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg|arg=="build-helper") {
        return justfit_rust_frontend::build_helpers::run(&args[1..]);
    }
    if let Some(result) = justfit_rust_frontend::cli::dispatch(&args) {
        if let Err(error)=result {eprintln!("justfit: error: {error}");std::process::exit(2);}
        return Ok(());
    }
    if args.len() == 2 && args[0] == "tokenizer-check" {
        justfit_rust_frontend::tokenizer_startup::check(Path::new(&args[1])).map_err(|e|e.1)?;
        println!("{}", json!({"passed":true}));
        return Ok(());
    }
    if args.len() == 2 && args[0] == "templates" {
        let front = Frontend::load(Path::new(&args[1]), "template-probe".into(), 8192).map_err(|e| e.1)?;
        println!("{}", serde_json::to_string(&front.chat_templates)?);
        return Ok(());
    }
    if args.len() == 4 && args[0] == "pdf-worker" {
        return justfit_rust_frontend::documents::worker(&args[1],&args[2],&args[3]).map_err(|e|e.1.into());
    }
    if args.len() == 1 && args[0] == "media-driver" {
        return justfit_rust_frontend::native_driver::run().map_err(|e| e.1.into());
    }
    if args.len() == 3 && args[0] == "native-cache-prepare" {
        let result = justfit_rust_frontend::ane_native::prepare_native_cache(
            Path::new(&args[1]),
            Path::new(&args[2]),
        )
        .map_err(|e| e.1)?;
        println!("{result}");
        return Ok(());
    }
    if args.len() == 2 && args[0] == "coreai-probe" {
        let result =
            justfit_rust_frontend::ane_native::probe(Path::new(&args[1])).map_err(|e| e.1)?;
        println!("{result}");
        return Ok(());
    }
    if args.len() == 5 && args[0] == "ane-worker" {
        let profile = justfit_rust_frontend::ane_native::Profile::open(Path::new(&args[1]))
            .map_err(|e| e.1)?;
        let _meter = justfit_rust_frontend::worker_meter::Meter::start(
            &profile
                .library
                .parent()
                .ok_or("bridge has no parent")?
                .join("attributed_memory.dylib"),
        )
        .map_err(|e| e.1)?;
        let grids: Vec<(usize, usize)> = serde_json::from_str(&args[4])?;
        let result = justfit_rust_frontend::ane_native::encode(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            &grids,
        )
        .map_err(|e| e.1)?;
        println!("{result}");
        return Ok(());
    }
    if args.len() == 6 && args[0] == "geometry-probe" {
        return justfit_rust_frontend::vision_math::probe(
            Path::new(&args[1]),
            Path::new(&args[2]),
            args[3].parse()?,
            args[4].parse()?,
            Path::new(&args[5]),
        )
        .map_err(|e| e.1.into());
    }
    if args.len() == 4 && args[0] == "image-probe" {
        if std::fs::metadata(&args[1])?.len() > 32 * 1024 * 1024 {
            return Err("image too large".into());
        }
        let image =
            justfit_rust_frontend::media::prepare(&std::fs::read(&args[1])?, args[3].parse()?)
                .map_err(|e| e.1)?;
        std::fs::write(&args[2], &image.pixels)?;
        println!(
            "{}",
            json!({"grid":[image.h,image.w],"digest_lo":image.lo,"digest_hi":image.hi})
        );
        return Ok(());
    }
    if args.len() == 2 && matches!(args[0].as_str(), "serve" | "serve-fd") {
        let bytes=if args[0]=="serve-fd" {
            use std::{io::Read,os::fd::FromRawFd};
            let fd:i32=args[1].parse()?;
            if fd<3{return Err("configuration descriptor must be at least 3".into())}
            // Take ownership so it cannot leak into native/media children.
            let mut input=unsafe{std::fs::File::from_raw_fd(fd)};
            let mut bytes=Vec::new();Read::by_ref(&mut input).take(1024*1024+1).read_to_end(&mut bytes)?;
            if bytes.len()>1024*1024{return Err("configuration exceeds 1MiB".into())}bytes
        }else{std::fs::read(&args[1])?};
        let config = serde_json::from_slice(&bytes)?;
        return tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .build()?
            .block_on(justfit_rust_frontend::http::serve(config));
    }
    if args.len() != 4 || args[0] != "jsonl" {
        return Err("usage: justfit-rs jsonl TOKENIZER_DIR MODEL CONTEXT".into());
    }
    let frontend =
        Frontend::load(Path::new(&args[1]), args[2].clone(), args[3].parse()?).map_err(|e| e.1)?;
    let mut stdout = io::stdout().lock();
    // Bounded line reader: no unbounded read_line allocation on untrusted input.
    let mut stdin = io::stdin().lock();
    loop {
        let mut line = Vec::new();
        loop {
            let buf = stdin.fill_buf()?;
            if buf.is_empty() {
                break;
            }
            let count = buf
                .iter()
                .position(|b| *b == b'\n')
                .map_or(buf.len(), |i| i + 1);
            if line.len() + count > 8 * 1024 * 1024 {
                return Err("JSONL line exceeds 8MiB".into());
            }
            line.extend_from_slice(&buf[..count]);
            stdin.consume(count);
            if line.last() == Some(&b'\n') {
                break;
            }
        }
        if line.is_empty() {
            break;
        }
        let result = json_body(&line)
            .and_then(|v| frontend.dispatch(v["op"].as_str().unwrap_or(""), &v["body"]));
        let response = match result {
            Ok(v) => json!({"status":200,"body":v}),
            Err(e) => json!({"status":e.0,"body":e.body()}),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
