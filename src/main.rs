mod cli;
mod context;
mod llm;
mod mcp;
mod session;
mod subagent;
mod tool;
use clap::Parser;
use cli::Cli;
use context::ContextManager;
use dotenv::dotenv;
use llm::create_llm;
use session::SessionManager;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tool::{BashTool, SubAgentTool, ToolRegistry};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // LLM Provider (openai, claude, minimax)
    #[arg(long, default_value = "minimax")]
    provider: String,

    // Model name
    #[arg(long, default_value = "MiniMax-M2.1")]
    model: String,

    // API Key (optional, can use env var)
    #[arg(long)]
    api_key: Option<String>,

    // API URL (optional, defaults to provider specific URL)
    #[arg(long, default_value = "https://api.minimaxi.com/anthropic/v1/messages")]
    api_url: Option<String>,

    // Session storage directory
    #[arg(long, default_value = "__sessions")]
    session_dir: String,

    // Maximum agent loops
    #[arg(long, default_value_t = 50)]
    max_loops: usize,

    // Log level (error, warn, info,debug, trace)
    #[arg(long, default_value = "debug")]
    log_level: String,

    // Log file path
    #[arg(long, default_value = "mini-agent.log")]
    log_file: String,

    // MCP config file path
    #[arg(long, default_value = "src/mcp/mcp.json")]
    mcp_config: String,

    // Disable MCP tools
    #[arg(long, default_value_t = false)]
    disable_mcp: bool,

    // Context max tokens
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
}

use env_logger::Builder;
use log::LevelFilter;
use std::fs::OpenOptions;
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let args = Args::parse();

    // Initialize logging with default info level if RUST_LOG is not set or empty
    let mut builder = Builder::from_default_env();
    builder.format(|buf, record| {
        use std::io::Write;
        let file = record.file().unwrap_or("unknown");
        let line = record
            .line()
            .map(|l| l.to_string())
            .unwrap_or_else(|| "-".to_string());
        let ts = buf.timestamp_millis();
        writeln!(
            buf,
            "{} [{}] {}:{} - {}",
            ts,
            record.level(),
            file,
            line,
            record.args()
        )
    });
    let rust_log = std::env::var("RUST_LOG").ok();
    if rust_log
        .as_deref()
        .map(|s| s.trim())
        .unwrap_or("")
        .is_empty()
    {
        let level = LevelFilter::from_str(&args.log_level).unwrap_or(LevelFilter::Debug);
        builder.filter(None, level);
    }
    // Always suppress rustyline logs to avoid noise in CLI
    builder.filter(Some("rustyline"), LevelFilter::Warn);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.log_file)?;
    builder.target(env_logger::Target::Pipe(Box::new(log_file)));
    builder.init();

    let api_key = args
        .api_key
        .as_ref()
        .cloned()
        .or_else(|| match args.provider.to_lowercase().as_str() {
            "openai" => std::env::var("OPENAI_API_KEY").ok(),
            "claude" => std::env::var("ANTHROPIC_API_KEY").ok(),
            "minimax" => std::env::var("MINIMAX_API_KEY").ok(),
            _ => None,
        })
        .expect("API Key must be provided via --api-key or env var (e.g. ANTHROPIC_API_KEY)");

    // Initialize components
    let llm = create_llm(&args.provider, &args.model, &api_key, args.api_url)?;
    let llm: Arc<dyn llm::LLM> = Arc::from(llm);

    let mut context = ContextManager::new(args.max_tokens); // Max tokens
    context.set_llm(llm.clone()); // Enable compression with LLM
    let session_manager = SessionManager::new(PathBuf::from(&args.session_dir));

    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(BashTool));
    tool_registry.register(Arc::new(SubAgentTool));
    // Register other tools or MCP tools here
    if !args.disable_mcp
        && let Err(e) = mcp::register_mcp_tools(&mut tool_registry, &args.mcp_config).await
    {
        log::error!("MCP tool registration failed: {}", e);
    }

    let mut ui = Cli::new(context, session_manager, tool_registry, llm, args.max_loops);

    ui.run().await?;
    Ok(())
}
