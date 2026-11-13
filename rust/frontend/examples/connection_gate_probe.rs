use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, Request, State},
    response::{IntoResponse, Response},
    Router,
};
use justfit_rust_frontend::http::connections::{ready_bytes, Gate, Info, Listener};
use std::{
    convert::Infallible,
    io::{self, BufRead},
    sync::Arc,
    time::Duration,
};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    justfit_rust_frontend::http::connections::descriptor_limit();
    if std::env::args().nth(1).as_deref() == Some("heads") {
        for line in io::stdin().lock().lines() {
            let bytes: Vec<u8> = serde_json::from_str(&line?)?;
            println!("{}", ready_bytes(&bytes));
        }
        return Ok(());
    }
    let tcp = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = tcp.local_addr()?.port();
    let (listener, gate) = Listener::new(tcp, 2)?;
    async fn route(State(gate): State<Arc<Gate>>, request: Request) -> Response {
        if let Some(info) = request.extensions().get::<ConnectInfo<Info>>() {
            let length = request
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            info.0.body(length, Duration::from_secs(30));
        }
        let mut response = match request.uri().path() {
            "/hold" => Response::new(Body::from_stream(tokio_stream::pending::<
                Result<Bytes, Infallible>,
            >())),
            "/reject" => (axum::http::StatusCode::PAYLOAD_TOO_LARGE, "rejected").into_response(),
            "/status" => gate.snapshot().to_string().into_response(),
            _ => "ok".into_response(),
        };
        response
            .headers_mut()
            .insert("connection", "close".parse().unwrap());
        response
    }
    println!(
        "{}",
        serde_json::json!({"port":port,"pid":std::process::id()})
    );
    axum::serve(
        listener,
        Router::new()
            .fallback(route)
            .with_state(gate)
            .into_make_service_with_connect_info::<Info>(),
    )
    .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.unwrap() })
    .await?;
    Ok(())
}
