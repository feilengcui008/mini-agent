use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const PROTOCOL_VERSION: &str = "2024-11-05";

#[tokio::main]
async fn main() -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");

        match method {
            "initialize" => {
                if let Some(id) = id {
                    let result = json!({
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "google-mcp",
                            "version": "0.1.0"
                        }
                    });
                    write_response(&mut stdout, id, Ok(result)).await?;
                }
            }
            "notifications/initialized" => {
                // no-op
            }
            "tools/list" => {
                if let Some(id) = id {
                    let result = json!({
                        "tools": [
                            {
                                "name": "google_search",
                                "description": "Search Google via Custom Search API",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Search query"
                                        },
                                        "num": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 10,
                                            "default": 5,
                                            "description": "Number of results (1-10)"
                                        },
                                        "safe": {
                                            "type": "boolean",
                                            "default": true,
                                            "description": "Enable safe search"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            }
                        ]
                    });
                    write_response(&mut stdout, id, Ok(result)).await?;
                }
            }
            "tools/call" => {
                if let Some(id) = id {
                    let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
                    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if name != "google_search" {
                        write_response(
                            &mut stdout,
                            id,
                            Err(json!({"code": -32601, "message": "Tool not found"})),
                        )
                        .await?;
                        continue;
                    }
                    let args = params
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let result = match handle_google_search(args).await {
                        Ok(text) => json!({
                            "content": [
                                { "type": "text", "text": text }
                            ],
                            "isError": false
                        }),
                        Err(e) => json!({
                            "content": [
                                { "type": "text", "text": format!("Error: {}", e) }
                            ],
                            "isError": true
                        }),
                    };
                    write_response(&mut stdout, id, Ok(result)).await?;
                }
            }
            _ => {
                if let Some(id) = id {
                    write_response(
                        &mut stdout,
                        id,
                        Err(json!({"code": -32601, "message": "Method not found"})),
                    )
                    .await?;
                }
            }
        }
    }

    Ok(())
}

async fn write_response(
    stdout: &mut tokio::io::Stdout,
    id: Value,
    result: std::result::Result<Value, Value>,
) -> Result<()> {
    let payload = match result {
        Ok(res) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": res
        }),
        Err(err) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": err
        }),
    };
    let line = format!("{}\n", payload);
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

async fn handle_google_search(args: Value) -> Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .context("Missing 'query'")?;
    let num = args.get("num").and_then(|v| v.as_u64()).unwrap_or(5);
    let num = num.clamp(1, 10);
    let safe = args.get("safe").and_then(|v| v.as_bool()).unwrap_or(true);

    let api_key = std::env::var("GOOGLE_API_KEY")
        .context("GOOGLE_API_KEY is required for Google Custom Search")?;
    let cx = std::env::var("GOOGLE_CX")
        .or_else(|_| std::env::var("GOOGLE_CSE_ID"))
        .context("GOOGLE_CX (or GOOGLE_CSE_ID) is required for Google Custom Search")?;

    let client = reqwest::Client::new();
    let url = "https://www.googleapis.com/customsearch/v1";
    let safe_param = if safe { "active" } else { "off" };

    let url = reqwest::Url::parse_with_params(
        url,
        &[
            ("key", api_key.as_str()),
            ("cx", cx.as_str()),
            ("q", query),
            ("num", &num.to_string()),
            ("safe", safe_param),
        ],
    )?;

    let resp = client.get(url).send().await?.error_for_status()?;

    let body: Value = resp.json().await?;
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if items.is_empty() {
        return Ok("No results.".to_string());
    }

    let mut lines = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        let title = item
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled");
        let link = item.get("link").and_then(|v| v.as_str()).unwrap_or("");
        let snippet = item.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
        lines.push(format!("{}. {}\\n{}\\n{}", idx + 1, title, link, snippet));
    }

    Ok(lines.join("\\n\\n"))
}
