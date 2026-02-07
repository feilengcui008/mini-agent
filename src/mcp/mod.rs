use crate::tool::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use log::{debug, error, info, warn};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Deserialize)]
pub struct McpConfig {
    pub servers: Vec<McpServerConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct McpToolInfo {
    name: String,
    description: String,
    input_schema: Value,
}

pub async fn register_mcp_tools(
    tool_registry: &mut crate::tool::ToolRegistry,
    config_path: &str,
) -> Result<()> {
    let path = Path::new(config_path);
    if !path.exists() {
        info!("MCP config not found at {}", config_path);
        return Ok(());
    }

    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read MCP config: {}", config_path))?;
    let cfg: McpConfig = serde_json::from_str(&content)
        .with_context(|| format!("Invalid MCP config JSON: {}", config_path))?;

    for server in cfg.servers {
        match McpClientHandle::connect(server.clone()).await {
            Ok(handle) => match handle.list_tools().await {
                Ok(tools) => {
                    for tool in tools {
                        let wrapper = McpTool::new(server.name.clone(), tool, handle.clone());
                        tool_registry.register(Arc::new(wrapper));
                    }
                }
                Err(e) => {
                    error!("MCP server '{}' list tools failed: {}", server.name, e);
                }
            },
            Err(e) => {
                error!("MCP server '{}' connect failed: {}", server.name, e);
            }
        }
    }

    Ok(())
}

#[derive(Clone)]
struct McpClientHandle {
    inner: Arc<Mutex<McpClientInner>>,
}

impl McpClientHandle {
    async fn connect(cfg: McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::inherit());
        if !cfg.env.is_empty() {
            cmd.envs(&cfg.env);
        }

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to spawn MCP server '{}' with command '{}'",
                cfg.name, cfg.command
            )
        })?;

        let stdin = child
            .stdin
            .take()
            .context("MCP server stdin not available")?;
        let stdout = child
            .stdout
            .take()
            .context("MCP server stdout not available")?;

        let mut inner = McpClientInner {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            server_name: cfg.name.clone(),
        };

        inner.initialize().await?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let mut inner = self.inner.lock().await;
        inner.list_tools().await
    }

    async fn call_tool(&self, name: &str, args: Value) -> Result<String> {
        let mut inner = self.inner.lock().await;
        inner.call_tool(name, args).await
    }
}

struct McpClientInner {
    _child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
    next_id: u64,
    server_name: String,
}

impl McpClientInner {
    async fn initialize(&mut self) -> Result<()> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "mini-agent",
                "version": "0.1.0"
            }
        });
        let _ = self.send_request("initialize", params).await?;
        self.send_notification("notifications/initialized", json!({}))
            .await?;
        Ok(())
    }

    async fn list_tools(&mut self) -> Result<Vec<McpToolInfo>> {
        let res = self.send_request("tools/list", json!({})).await?;
        let tools = res
            .get("tools")
            .and_then(|v| v.as_array())
            .context("MCP tools/list response missing tools")?;

        let mut out = Vec::new();
        for t in tools {
            let name = t
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if name.is_empty() {
                continue;
            }
            let description = t
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = t.get("inputSchema").cloned().unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {}
                })
            });
            out.push(McpToolInfo {
                name,
                description,
                input_schema,
            });
        }
        Ok(out)
    }

    async fn call_tool(&mut self, name: &str, args: Value) -> Result<String> {
        let params = json!({
            "name": name,
            "arguments": args
        });
        let res = self.send_request("tools/call", params).await?;
        let is_error = res
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let content = res.get("content").and_then(|v| v.as_array()).cloned();

        let mut texts = Vec::new();
        if let Some(items) = content {
            for item in items {
                if item.get("type").and_then(|v| v.as_str()) == Some("text")
                    && let Some(text) = item.get("text").and_then(|v| v.as_str())
                {
                    texts.push(text.to_string());
                }
            }
        }

        let joined = if texts.is_empty() {
            serde_json::to_string_pretty(&res).unwrap_or_else(|_| res.to_string())
        } else {
            texts.join("\n")
        };

        if is_error {
            Err(anyhow::anyhow!("MCP tool error: {}", joined))
        } else {
            Ok(joined)
        }
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        let line = format!("{}\n", req);
        self.stdin
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("MCP write failed (server: {})", self.server_name))?;
        self.stdin
            .flush()
            .await
            .with_context(|| format!("MCP flush failed (server: {})", self.server_name))?;

        loop {
            let mut buf = String::new();
            let n = self.stdout.read_line(&mut buf).await?;
            if n == 0 {
                return Err(anyhow::anyhow!(
                    "MCP server '{}' closed stdout",
                    self.server_name
                ));
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            let msg: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "MCP parse error from '{}': {} -> {}",
                        self.server_name, trimmed, e
                    );
                    continue;
                }
            };
            if msg.get("id").and_then(|v| v.as_u64()) != Some(id) {
                debug!(
                    "MCP message ignored (server: {}): {}",
                    self.server_name, msg
                );
                continue;
            }
            if let Some(err) = msg.get("error") {
                return Err(anyhow::anyhow!("MCP error: {}", err));
            }
            if let Some(result) = msg.get("result") {
                return Ok(result.clone());
            }
            return Err(anyhow::anyhow!(
                "MCP response missing result for '{}'",
                method
            ));
        }
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        let req = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let line = format!("{}\n", req);
        self.stdin
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("MCP notify write failed (server: {})", self.server_name))?;
        self.stdin
            .flush()
            .await
            .with_context(|| format!("MCP notify flush failed (server: {})", self.server_name))?;
        Ok(())
    }
}

struct McpTool {
    full_name: String,
    full_description: String,
    tool: McpToolInfo,
    client: McpClientHandle,
}

impl McpTool {
    fn new(server_name: String, tool: McpToolInfo, client: McpClientHandle) -> Self {
        let full_name = format!("mcp.{}.{}", server_name, tool.name.clone());
        let full_description = format!("[MCP:{}] {}", server_name, tool.description.clone());
        Self {
            full_name,
            full_description,
            tool,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.full_name
    }

    fn description(&self) -> &str {
        &self.full_description
    }

    fn schema(&self) -> Value {
        self.tool.input_schema.clone()
    }

    async fn call(&self, args: Value) -> Result<String> {
        self.client.call_tool(&self.tool.name, args).await
    }
}
