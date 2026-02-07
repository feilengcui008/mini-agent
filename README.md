# Introduction

A very simple agent for learning, support cli ui, tool use, session management, context compression, subagent etc.

## MCP

This project supports MCP (Model Context Protocol) tools via a simple stdio client.

1. Build the Google MCP server binary:

```bash
cargo build --bin google_mcp
```

2. Create `mcp.json` in the project root:

```json
{
  "servers": [
    {
      "name": "google",
      "command": "target/debug/google_mcp",
      "args": [],
      "env": {
        "GOOGLE_API_KEY": "your_api_key",
        "GOOGLE_CX": "your_cse_id"
      }
    }
  ]
}
```

3. Run the agent (MCP enabled by default):

```bash
cargo run
```

You can disable MCP with `--disable-mcp` or specify a different config via `--mcp-config`.
