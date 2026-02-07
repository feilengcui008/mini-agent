use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> Value;
    async fn call(&self, args: Value) -> Result<String>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<_> = self.tools.values().cloned().collect();
        tools.sort_by(|a, b| a.name().cmp(b.name()));
        tools
    }

    pub fn generate_system_prompt(&self) -> String {
        let mut prompt = String::from("You are a helpful coding agent.\n\n");
        prompt.push_str("When you are finished, wrap the final answer in <final>...</final>.\n");
        prompt.push_str("If you need more steps and no tool call is required, continue until you are ready to finalize.\n\n");
        prompt.push_str(&self.generate_tool_instructions());
        prompt
    }

    pub fn generate_tool_instructions(&self) -> String {
        let mut prompt = String::from("You have access to the following tools:\n\n");
        for tool in self.list() {
            prompt.push_str(&format!("## {}: {}\n", tool.name(), tool.description()));
            prompt.push_str(&format!("Schema: {}\n\n", tool.schema()));
        }
        prompt.push_str("To use a tool, ONLY output a JSON block wrapped in <tool_code> tags. The JSON must be valid and directly deserializable. Do not double-encode JSON strings or escape quotes inside JSON values.\nExample:\n<tool_code>\n{\n  \"name\": \"bash\",\n  \"args\": {\n    \"command\": \"ls -la\"\n  }\n}\n</tool_code>\n");
        prompt.push_str("After the tool execution, you will receive the output. Then you can continue to answer the user's question.\n");
        prompt
    }
}

// Built-in Bash Tool
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: Value) -> Result<String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing command argument"))?;

        let output = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Ok(format!("Error: {}\nStdout: {}", stderr, stdout))
        }
    }
}

// SubAgentTool - Spawns a new subagent to handle a task
// The actual subagent execution requires SubAgentManager, which should be
// integrated at the CLI level. This tool provides the interface.
pub struct SubAgentTool;

#[async_trait]
impl Tool for SubAgentTool {
    fn name(&self) -> &str {
        "subagent"
    }

    fn description(&self) -> &str {
        "Spawn a new subagent to handle a specific task (parallel execution supported)"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description for the subagent"
                },
                "type": {
                    "type": "string",
                    "description": "SubAgent type: code, test, doc, analysis, or dynamic (default)"
                },
                "max_loops": {
                    "type": "integer",
                    "description": "Maximum loop iterations (default: 20)"
                }
            },
            "required": ["task"]
        })
    }

    async fn call(&self, args: Value) -> Result<String> {
        let task = args["task"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'task' argument"))?;
        let agent_type = args["type"].as_str().unwrap_or("dynamic");
        let max_loops = args["max_loops"].as_u64().unwrap_or(20) as usize;

        // Note: Full subagent execution requires integration with SubAgentManager
        // at the CLI level. This tool provides the interface specification.
        Ok(format!(
            "SubAgent spawned: type='{}', task='{}', max_loops={}. \
             (SubAgent execution requires SubAgentManager integration)",
            agent_type, task, max_loops
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_tool() {
        let tool = BashTool;
        let args = serde_json::json!({
            "command": "echo 'Hello Tool'"
        });
        let result = tool.call(args).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().trim(), "Hello Tool");
    }

    #[test]
    fn test_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(BashTool));
        assert!(registry.get("bash").is_some());
    }
}
